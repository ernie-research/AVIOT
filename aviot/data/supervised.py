"""Video-only supervised data path used to train AVIOT."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import yaml

from aviot.data.question_extraction import extract_aviot_query_text, find_token_range_for_text
from aviot.data.video import VideoSamplingConfig, decode_video, preprocess_video
from aviot.mm_utils import DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, VIDEO_TOKEN_INDEX, tokenizer_video_token


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


@dataclass(frozen=True)
class AnnotationSource:
    path: Path
    strategy: str = "all"


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield value
        return
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON list")
        for record in value:
            if not isinstance(record, dict):
                raise ValueError(f"every item in {path} must be a JSON object")
            yield record
        return
    raise ValueError(f"unsupported annotation extension: {path.suffix}")


def read_annotation_sources(path: str | Path) -> list[AnnotationSource]:
    """Resolve a JSON/JSONL file or a YAML dataset manifest."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"annotation file not found: {manifest}")
    if manifest.suffix not in {".yaml", ".yml"}:
        return [AnnotationSource(manifest)]
    value = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    datasets = value.get("datasets") if isinstance(value, Mapping) else None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset YAML must contain a non-empty 'datasets' list")
    result: list[AnnotationSource] = []
    for index, item in enumerate(datasets):
        if not isinstance(item, Mapping) or not item.get("json_path"):
            raise ValueError(f"dataset entry {index} must define json_path")
        item_path = Path(str(item["json_path"])).expanduser()
        if not item_path.is_absolute():
            item_path = manifest.parent / item_path
        result.append(
            AnnotationSource(
                item_path.resolve(),
                str(item.get("sampling_strategy", "all")),
            )
        )
    return result


def _sample_records(
    records: list[dict[str, Any]],
    strategy: str,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    strategy = strategy.strip().lower()
    if strategy == "all":
        return records
    if ":" not in strategy:
        raise ValueError(f"invalid sampling strategy: {strategy!r}")
    mode, amount_text = strategy.split(":", 1)
    if amount_text.endswith("%"):
        amount = math.ceil(float(amount_text[:-1]) * len(records) / 100.0)
    else:
        amount = int(amount_text)
    amount = max(0, min(amount, len(records)))
    if mode == "first":
        return records[:amount]
    if mode == "end":
        return records[-amount:] if amount else []
    if mode == "random":
        indices = list(range(len(records)))
        random.Random(seed).shuffle(indices)
        return [records[index] for index in indices[:amount]]
    raise ValueError(f"unsupported sampling mode: {mode!r}")


def load_annotations(path: str | Path, *, seed: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(read_annotation_sources(path)):
        current = list(_iter_json_records(source.path))
        records.extend(
            _sample_records(current, source.strategy, seed=seed + source_index)
        )
    if not records:
        raise ValueError("the training annotations contain no examples")
    return records


def _turn_fields(turn: Mapping[str, Any]) -> tuple[str, str]:
    role_value = turn.get("from", turn.get("role"))
    content_value = turn.get("value", turn.get("content"))
    role = ROLE_MAP.get(str(role_value).lower())
    if role is None:
        raise ValueError(f"unsupported conversation role: {role_value!r}")
    if not isinstance(content_value, str):
        raise ValueError("conversation content must be a string")
    return role, content_value


def build_time_instruction(sample) -> str:
    return (
        f"The video lasts for {sample.duration:.2f} seconds, and "
        f"{sample.num_frames} frames are uniformly sampled from it. "
        f"These frames are located at {sample.frame_time_text}."
        "Please answer the following questions related to this video."
    )


def prepare_conversations(
    conversations: Sequence[Mapping[str, Any]],
    sample,
) -> list[dict[str, str]]:
    if not conversations:
        raise ValueError("a training example must contain at least one conversation turn")
    result: list[dict[str, str]] = []
    for index, raw_turn in enumerate(conversations):
        role, content = _turn_fields(raw_turn)
        if index == 0:
            if role != "user":
                raise ValueError("the first conversation turn must be from the user")
            content = content.replace(DEFAULT_VIDEO_TOKEN, "").strip()
            content = f"{DEFAULT_VIDEO_TOKEN}\n{build_time_instruction(sample)}\n{content}"
        elif DEFAULT_VIDEO_TOKEN in content:
            raise ValueError("the video placeholder may appear only in the first user turn")
        result.append({"role": role, "content": content})
    return result


def encode_chatml(
    conversations: Sequence[Mapping[str, str]],
    tokenizer,
    *,
    system_message: str = "You are a helpful assistant.",
) -> dict[str, torch.Tensor]:
    """Encode Qwen ChatML and mask all non-assistant text."""

    input_ids: list[int] = []
    labels: list[int] = []
    ranges: list[list[int]] = []

    def append_turn(role: str, content: str, supervise: bool) -> None:
        segment = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        segment_ids = tokenizer_video_token(segment, tokenizer)
        offset = len(input_ids)
        input_ids.extend(segment_ids)
        labels.extend(segment_ids if supervise else [IGNORE_INDEX] * len(segment_ids))
        if role == "user":
            query = extract_aviot_query_text(content)
            if query:
                start, end = find_token_range_for_text(
                    segment_ids,
                    query,
                    lambda value: tokenizer(value).input_ids,
                )
                if start >= 0:
                    ranges.append([offset + start, offset + end])

    append_turn("system", system_message, False)
    for turn in conversations:
        append_turn(
            str(turn["role"]),
            str(turn["content"]),
            str(turn["role"]) == "assistant",
        )
    if input_ids.count(VIDEO_TOKEN_INDEX) != 1:
        raise ValueError("each training example must contain exactly one video placeholder")
    if not ranges:
        ranges = [[-1, -1]]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "question_token_ranges": torch.tensor(ranges, dtype=torch.long),
    }


class VideoSupervisedDataset(Dataset):
    """Lazy video decoding with deterministic AVIOT sampling."""

    def __init__(
        self,
        *,
        annotations: str | Path,
        video_root: str | Path,
        tokenizer,
        video_processor,
        sampling: VideoSamplingConfig,
        seed: int = 0,
    ) -> None:
        self.records = load_annotations(annotations, seed=seed)
        self.video_root = Path(video_root).expanduser().resolve()
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.sampling = sampling

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video_value = record.get("video")
        if not isinstance(video_value, str) or not video_value:
            raise ValueError(f"training example {index} has no video path")
        path = Path(video_value).expanduser()
        if not path.is_absolute():
            path = self.video_root / path
        sample = decode_video(path, self.sampling)
        encoded = encode_chatml(
            prepare_conversations(record.get("conversations", ()), sample),
            self.tokenizer,
        )
        encoded["videos"] = preprocess_video(sample, self.video_processor)
        return encoded


@dataclass
class VideoSupervisedCollator:
    tokenizer: Any
    model_max_length: int

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("AVIOT training requires right-padding for question ranges")
        limit = int(self.model_max_length)
        input_rows = [item["input_ids"][:limit] for item in examples]
        label_rows = [item["labels"][:limit] for item in examples]
        if any(int((row == VIDEO_TOKEN_INDEX).sum()) != 1 for row in input_rows):
            raise RuntimeError("text truncation removed or duplicated a video placeholder")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("the tokenizer must define a pad or EOS token")
        input_ids = pad_sequence(input_rows, batch_first=True, padding_value=int(pad_id))
        labels = pad_sequence(label_rows, batch_first=True, padding_value=IGNORE_INDEX)

        range_rows = []
        for item in examples:
            kept = []
            for start, end in item["question_token_ranges"].view(-1, 2).tolist():
                if 0 <= start < end and start < limit:
                    kept.append([start, min(end, limit)])
            range_rows.append(torch.tensor(kept or [[-1, -1]], dtype=torch.long))
        maximum_ranges = max(row.shape[0] for row in range_rows)
        question_ranges = torch.full(
            (len(range_rows), maximum_ranges, 2), -1, dtype=torch.long
        )
        for index, row in enumerate(range_rows):
            question_ranges[index, : row.shape[0]] = row

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(int(pad_id)),
            "question_token_ranges": question_ranges,
            "videos": [item["videos"] for item in examples],
        }


__all__ = [
    "AnnotationSource",
    "VideoSupervisedCollator",
    "VideoSupervisedDataset",
    "build_time_instruction",
    "encode_chatml",
    "load_annotations",
    "prepare_conversations",
    "read_annotation_sources",
]
