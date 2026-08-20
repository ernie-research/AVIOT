"""Tokenization helpers for the single-video multimodal prompt."""

from __future__ import annotations

from typing import Any

import torch


IGNORE_INDEX = -100
VIDEO_TOKEN_INDEX = -200
DEFAULT_VIDEO_TOKEN = "<image>"


def tokenizer_video_token(
    prompt: str,
    tokenizer: Any,
    video_token_index: int = VIDEO_TOKEN_INDEX,
    return_tensors: str | None = None,
) -> list[int] | torch.Tensor:
    """Tokenize a prompt while replacing the video placeholder with a sentinel."""

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_VIDEO_TOKEN)]
    input_ids: list[int] = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for chunk_index, chunk in enumerate(prompt_chunks):
        if chunk_index:
            input_ids.append(video_token_index)
        input_ids.extend(chunk[offset:])
    if return_tensors is None:
        return input_ids
    if return_tensors != "pt":
        raise ValueError(f"unsupported return_tensors={return_tensors!r}")
    return torch.tensor(input_ids, dtype=torch.long)


__all__ = [
    "DEFAULT_VIDEO_TOKEN",
    "IGNORE_INDEX",
    "VIDEO_TOKEN_INDEX",
    "tokenizer_video_token",
]
