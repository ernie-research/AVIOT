"""High-level video question answering with AVIOT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
from typing import Any, Sequence

import torch

from .data.question_extraction import (
    extract_aviot_query_text,
    find_token_range_for_text,
)
from .data.video import (
    VideoSample,
    VideoSamplingConfig,
    decode_video,
    preprocess_video,
)
from .mm_utils import DEFAULT_VIDEO_TOKEN, VIDEO_TOKEN_INDEX, tokenizer_video_token
from .modeling.loader import load_pretrained_model, set_inference_ratio


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float | None = None
    num_beams: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")


@dataclass(frozen=True)
class AVIOTPrediction:
    text: str
    prompt: str
    question: str
    ratio: float
    input_frames: int
    target_supports: int
    sampling: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AVIOTGenerator:
    """Load one AVIOT checkpoint and answer questions about local videos."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        local_files_only: bool = False,
    ) -> None:
        self.tokenizer, self.model = load_pretrained_model(
            checkpoint,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        self.video_processor = self.model.get_vision_tower().video_processor

    @torch.inference_mode()
    def answer(
        self,
        video: str | Path,
        question: str,
        *,
        ratio: float = 4.0,
        sampling: VideoSamplingConfig = VideoSamplingConfig(),
        generation: GenerationSettings = GenerationSettings(),
        seed: int = 0,
    ) -> AVIOTPrediction:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        set_inference_ratio(self.model, ratio)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        sample = decode_video(video, sampling)
        vision_tower = self.model.get_vision_tower()
        video_tensor = preprocess_video(
            sample,
            self.video_processor,
            device=vision_tower.device,
            dtype=vision_tower.dtype,
        )
        user_text = format_video_question(question, sample)
        prompt = build_qwen_prompt(user_text)
        input_device = self.model.get_input_embeddings().weight.device
        input_ids = tokenizer_video_token(
            prompt,
            self.tokenizer,
            VIDEO_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(input_device)
        question_ranges = build_question_token_ranges(
            user_text,
            self.tokenizer,
            input_ids,
        ).to(input_device)

        do_sample = generation.temperature > 0
        generate_kwargs: dict[str, Any] = {
            "videos": [video_tensor],
            "question_token_ranges": question_ranges,
            "do_sample": do_sample,
            "num_beams": generation.num_beams,
            "max_new_tokens": generation.max_new_tokens,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = generation.temperature
            if generation.top_p is not None:
                generate_kwargs["top_p"] = generation.top_p
        stop_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [
            token_id
            for token_id in (self.tokenizer.eos_token_id, stop_id)
            if token_id is not None and token_id >= 0
        ]
        if eos_ids:
            generate_kwargs["eos_token_id"] = sorted(set(eos_ids))

        output_ids = self.model.generate(input_ids, **generate_kwargs)
        text = decode_completion(self.tokenizer, input_ids, output_ids)
        text = text.split("<|im_end|>", 1)[0].strip()
        target_supports = min(sample.num_frames, max(1, math.ceil(sample.num_frames / float(ratio))))
        return AVIOTPrediction(
            text=text,
            prompt=prompt,
            question=question,
            ratio=float(ratio),
            input_frames=sample.num_frames,
            target_supports=target_supports,
            sampling={
                "fps": sampling.fps,
                "max_frames": sampling.max_frames,
                "force_uniform": sampling.force_uniform,
                "backend": sample.backend,
                "native_fps": sample.native_fps,
                "duration": sample.duration,
                "physical_frame_count": sample.physical_frame_count,
                "frame_indices": list(sample.frame_indices),
                "frame_times": list(sample.frame_times),
                **sample.metadata,
            },
        )


def format_video_question(question: str, sample: VideoSample) -> str:
    """Reproduce the time instruction used for AVIOT training."""

    instruction = (
        f"The video lasts for {sample.duration:.2f} seconds, and "
        f"{sample.num_frames} frames are uniformly sampled from it. "
        f"These frames are located at {sample.frame_time_text}."
        "Please answer the following questions related to this video."
    )
    return f"{DEFAULT_VIDEO_TOKEN}\n{instruction}\n{question.strip()}"

def build_qwen_prompt(user_text: str) -> str:
    """Build the Qwen2 ChatML prompt used by the final model."""

    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_question_token_ranges(
    user_text: str,
    tokenizer: Any,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Locate the semantic question span consumed by question conditioning."""

    ranges = torch.full((1, 1, 2), -1, dtype=torch.long)
    question_text = extract_aviot_query_text(user_text)
    if not question_text or input_ids.numel() == 0:
        return ranges
    start, end = find_token_range_for_text(
        input_ids[0].detach().cpu().tolist(),
        question_text,
        lambda text: tokenizer(text).input_ids,
    )
    if start >= 0:
        ranges[0, 0] = torch.tensor([start, end], dtype=torch.long)
    return ranges


def decode_completion(
    tokenizer: Any,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor | Any,
) -> str:
    """Handle both completion-only and prompt-prefixed generation outputs."""

    if not isinstance(output_ids, torch.Tensor):
        output_ids = getattr(output_ids, "sequences", None)
    if not isinstance(output_ids, torch.Tensor):
        raise TypeError("generate output does not contain token sequences")
    if output_ids.ndim != 2 or output_ids.shape[0] != 1:
        raise ValueError(f"expected generated ids [1,T], got {tuple(output_ids.shape)}")
    row = output_ids[0]
    prompt = input_ids[0]
    if row.numel() >= prompt.numel() and torch.equal(
        row[: prompt.numel()].to(prompt.device),
        prompt,
    ):
        row = row[prompt.numel() :]
    return tokenizer.decode(row, skip_special_tokens=True).strip()


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Answer one question about a video with AVIOT.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--ratio", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=224)
    parser.add_argument("--force-uniform", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generator = AVIOTGenerator(
        args.checkpoint,
        device=args.device,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    prediction = generator.answer(
        args.video,
        args.question,
        ratio=args.ratio,
        sampling=VideoSamplingConfig(
            fps=args.fps,
            max_frames=args.max_frames,
            force_uniform=args.force_uniform,
        ),
        generation=GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
        ),
        seed=args.seed,
    )
    import json

    print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
