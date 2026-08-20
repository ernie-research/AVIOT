"""Command-line evaluation for generic video-question JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..data.video import VideoSamplingConfig
from ..inference import AVIOTGenerator, GenerationSettings


@dataclass(frozen=True)
class EvaluationExample:
    """One evaluation item accepted by the public evaluator."""

    sample_id: str
    video: str
    question: str
    answer: str | None = None
    choices: tuple[str, ...] = ()


def read_examples(path: str | Path) -> Iterator[EvaluationExample]:
    """Read a JSON array, a JSON object containing ``items``, or JSONL."""

    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"evaluation file not found: {input_path}")
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                parsed = parsed.get("items")
            if not isinstance(parsed, list):
                raise ValueError("JSON evaluation input must be a list or an object with an 'items' list")
            for index, item in enumerate(parsed):
                yield parse_example(item, index)
            return
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {index}: {exc}") from exc
        yield parse_example(item, index)


def parse_example(item: Any, index: int) -> EvaluationExample:
    if not isinstance(item, dict):
        raise ValueError(f"evaluation item {index} must be a JSON object")
    sample_id = str(item.get("id", index))
    video = item.get("video")
    question = item.get("question")
    if not isinstance(video, str) or not video.strip():
        raise ValueError(f"evaluation item {sample_id!r} has no non-empty 'video' field")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"evaluation item {sample_id!r} has no non-empty 'question' field")
    answer = item.get("answer")
    if answer is not None:
        answer = str(answer)
    raw_choices = item.get("choices", ())
    if raw_choices is None:
        raw_choices = ()
    if not isinstance(raw_choices, (list, tuple)):
        raise ValueError(f"evaluation item {sample_id!r} field 'choices' must be a list")
    choices = tuple(str(choice) for choice in raw_choices)
    return EvaluationExample(
        sample_id=sample_id,
        video=video,
        question=question,
        answer=answer,
        choices=choices,
    )


def resolve_video_path(video: str, video_root: str | Path | None) -> Path:
    path = Path(video).expanduser()
    if not path.is_absolute() and video_root is not None:
        path = Path(video_root).expanduser() / path
    return path


def normalize_answer(value: str) -> str:
    value = value.strip().casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value, flags=re.UNICODE)
    return value.strip()


def extract_choice(text: str, choices: Sequence[str] = ()) -> str:
    """Return a choice letter when present, otherwise the normalized text."""

    cleaned = text.strip()
    match = re.search(
        r"\b(?:answer|option|choice|final answer)\s*(?:is\s*)?[:\-]?\s*\(?([A-Z])\)?\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.match(r"^\s*\(?([A-Z])\)?(?:\s*[\.:\-]|\s*$)", cleaned, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if not choices or ord(letter) - ord("A") < len(choices):
            return letter
    normalized = normalize_answer(cleaned)
    for index, choice in enumerate(choices):
        if normalized == normalize_answer(choice):
            return chr(ord("A") + index)
    return normalized


def score_prediction(prediction: str, example: EvaluationExample, metric: str) -> bool | None:
    if example.answer is None or metric == "none":
        return None
    if metric == "exact_match":
        return normalize_answer(prediction) == normalize_answer(example.answer)
    if metric == "multiple_choice":
        return extract_choice(prediction, example.choices) == extract_choice(
            example.answer,
            example.choices,
        )
    raise ValueError(f"unsupported metric: {metric}")


def evaluate(
    generator: AVIOTGenerator,
    examples: Iterable[EvaluationExample],
    *,
    video_root: str | Path | None = None,
    ratio: float = 4.0,
    sampling: VideoSamplingConfig = VideoSamplingConfig(),
    generation: GenerationSettings = GenerationSettings(),
    metric: str = "none",
    seed: int = 0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference and optionally score predictions without creating caches."""

    if metric not in {"none", "exact_match", "multiple_choice"}:
        raise ValueError(f"unsupported metric: {metric}")
    output_handle = None
    if output_path is not None:
        output_handle = Path(output_path).expanduser().open("w", encoding="utf-8")
    total = 0
    correct = 0
    scored = 0
    try:
        for example in examples:
            prediction = generator.answer(
                resolve_video_path(example.video, video_root),
                example.question,
                ratio=ratio,
                sampling=sampling,
                generation=generation,
                seed=seed,
            )
            is_correct = score_prediction(prediction.text, example, metric)
            result = {
                "id": example.sample_id,
                "video": example.video,
                "question": example.question,
                "prediction": prediction.text,
                "answer": example.answer,
                "correct": is_correct,
                "ratio": prediction.ratio,
                "input_frames": prediction.input_frames,
                "target_supports": prediction.target_supports,
            }
            if output_handle is not None:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
            total += 1
            if is_correct is not None:
                scored += 1
                correct += int(is_correct)
    finally:
        if output_handle is not None:
            output_handle.close()
    summary: dict[str, Any] = {
        "total": total,
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored if scored else None),
        "metric": metric,
        "ratio": float(ratio),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AVIOT on JSON or JSONL video questions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSON array, object with 'items', or JSONL")
    parser.add_argument("--output", default=None, help="Optional JSONL prediction output")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--ratio", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=224)
    parser.add_argument("--force-uniform", action="store_true")
    parser.add_argument("--metric", choices=("none", "exact_match", "multiple_choice"), default="none")
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
    summary = evaluate(
        generator,
        read_examples(args.input),
        video_root=args.video_root,
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
        metric=args.metric,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
