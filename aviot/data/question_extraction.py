"""Shared AVIOT query extraction for training and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Sequence

from aviot.mm_utils import DEFAULT_VIDEO_TOKEN


_QUESTION_LABEL_RE = re.compile(r"^\s*(?:question|q|问题|提问)\s*[:：]", re.IGNORECASE)
_OPTIONS_LABEL_RE = re.compile(r"^\s*(?:options?|choices?|answer options?|选项|候选答案)\s*[:：]\s*$", re.IGNORECASE)
_OPTION_LINE_RE = re.compile(r"^\s*(?:\(?[A-Ha-h]\)?[\.\)]|[A-Ha-h]\s*[:：])\s+")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:answer|the\s+best\s+answer\s+is|final\s+answer|答案|正确答案)\s*[:：]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AVIOTQueryExtraction:
    query_text: str
    strategy: str
    dropped_lines: List[str]


def find_token_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last_start = len(haystack) - len(needle)
    for start in range(last_start + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return -1


def find_token_range_for_text(
    sequence: Sequence[int],
    text: str,
    encode_text: Callable[[str], Sequence[int]],
) -> tuple[int, int]:
    sequence_ids = list(sequence)
    for candidate in _text_boundary_variants(text):
        candidate_ids = list(encode_text(candidate))
        start = find_token_subsequence(sequence_ids, candidate_ids)
        if start >= 0:
            return start, start + len(candidate_ids)
    return _find_token_range_by_anchors(sequence_ids, text, encode_text)


def extract_aviot_query_text(text: str, video_token: str = DEFAULT_VIDEO_TOKEN) -> str:
    return extract_aviot_query(text, video_token=video_token).query_text


def extract_aviot_query(text: str, video_token: str = DEFAULT_VIDEO_TOKEN) -> AVIOTQueryExtraction:
    candidate = _post_video_text(text or "", video_token=video_token)
    raw_lines = [line.strip() for line in candidate.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return AVIOTQueryExtraction(query_text="", strategy="empty", dropped_lines=[])

    question_idx = _first_question_label_index(lines)
    if question_idx >= 0:
        query_lines, dropped = _collect_query_block(lines, question_idx)
        if query_lines:
            return AVIOTQueryExtraction(
                query_text="\n".join(query_lines).strip(),
                strategy="question_label_block",
                dropped_lines=dropped,
            )

    start_idx = _first_unlabeled_query_index(lines)
    if start_idx >= 0:
        query_lines, dropped = _collect_query_block(lines, start_idx)
        if query_lines:
            strategy = "unlabeled_mc_or_task_block" if _contains_option_line(query_lines) else "unlabeled_task"
            return AVIOTQueryExtraction(
                query_text="\n".join(query_lines).strip(),
                strategy=strategy,
                dropped_lines=dropped,
            )

    fallback_lines = [line for line in lines if not _is_drop_line(line)]
    if fallback_lines:
        return AVIOTQueryExtraction(
            query_text="\n".join(fallback_lines).strip(),
            strategy="filtered_fallback",
            dropped_lines=[line for line in lines if _is_drop_line(line)],
        )
    return AVIOTQueryExtraction(query_text=lines[-1].strip(), strategy="last_line_fallback", dropped_lines=lines[:-1])


def _post_video_text(text: str, video_token: str) -> str:
    if video_token and video_token in text:
        return text.split(video_token, 1)[1].strip()
    return text.strip()


def _text_boundary_variants(text: str) -> List[str]:
    stripped = (text or "").strip()
    bodies = [stripped, " " + stripped]
    if "\n" in stripped:
        bodies.append(stripped.replace("\n", "\r\n"))
        bodies.append(stripped.replace("\n", " \n"))
        bodies.append(stripped.replace(f"{DEFAULT_VIDEO_TOKEN}\n", f"{DEFAULT_VIDEO_TOKEN} \n"))
        bodies.append(stripped.replace(f"{DEFAULT_VIDEO_TOKEN}\n", f"{DEFAULT_VIDEO_TOKEN} \r\n"))

    variants: List[str] = []
    for body in bodies:
        variants.extend([body, body + "\n", body + "\n\n", body + " "])
    deduped: List[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _find_token_range_by_anchors(
    sequence: Sequence[int],
    text: str,
    encode_text: Callable[[str], Sequence[int]],
) -> tuple[int, int]:
    encoded_variants = [list(encode_text(candidate)) for candidate in _text_boundary_variants(text)]
    if not encoded_variants or max(len(item) for item in encoded_variants) < 4:
        return -1, -1

    for prefix_ids in encoded_variants:
        for prefix_size in _anchor_sizes(len(prefix_ids)):
            prefix = prefix_ids[:prefix_size]
            start = find_token_subsequence(sequence, prefix)
            if start < 0:
                continue
            for suffix_ids in encoded_variants:
                for suffix_size in _anchor_sizes(len(suffix_ids)):
                    suffix = suffix_ids[-suffix_size:]
                    suffix_start = _find_last_token_subsequence(sequence, suffix, min_start=start + prefix_size)
                    if suffix_start >= start:
                        return start, suffix_start + suffix_size
    return -1, -1


def _anchor_sizes(length: int) -> List[int]:
    sizes = [64, 48, 32, 24, 16, 12, 8, 6, 4]
    return [size for size in sizes if size <= length]


def _find_last_token_subsequence(haystack: Sequence[int], needle: Sequence[int], min_start: int = 0) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last_start = len(haystack) - len(needle)
    for start in range(last_start, max(min_start, 0) - 1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return -1


def _first_question_label_index(lines: Sequence[str]) -> int:
    for idx, line in enumerate(lines):
        if _QUESTION_LABEL_RE.match(line):
            return idx
    return -1


def _first_unlabeled_query_index(lines: Sequence[str]) -> int:
    for idx, line in enumerate(lines):
        if _is_drop_line(line) or _OPTIONS_LABEL_RE.match(line) or _OPTION_LINE_RE.match(line):
            continue
        return idx
    return -1


def _collect_query_block(lines: Sequence[str], start_idx: int) -> tuple[List[str], List[str]]:
    query_lines: List[str] = []
    dropped: List[str] = []
    for line in lines[start_idx:]:
        if _is_drop_line(line):
            dropped.append(line)
            if query_lines:
                break
            continue
        query_lines.append(line)

    while query_lines and _is_drop_line(query_lines[-1]):
        dropped.append(query_lines.pop())
    return query_lines, dropped


def _contains_option_line(lines: Sequence[str]) -> bool:
    return any(_OPTIONS_LABEL_RE.match(line) or _OPTION_LINE_RE.match(line) for line in lines)


def _is_drop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if _ANSWER_PREFIX_RE.match(stripped):
        return True
    if lower.startswith("prompt:") and (
        "select the best answer" in lower
        or "multiple-choice" in lower
        or "respond with only" in lower
        or "the best answer is" in lower
    ):
        return True
    if lower.startswith("select the best answer") and (
        "multiple-choice" in lower or "correct option" in lower or "based on the video" in lower
    ):
        return True
    if lower.startswith("the video lasts for ") and "frames" in lower and "sampled" in lower:
        return True
    if lower.startswith("these frames are located at"):
        return True
    if "please answer the following questions related to this video" in lower:
        return True
    if _is_answer_instruction(lower):
        return True
    return False


def _is_answer_instruction(lower_line: str) -> bool:
    if lower_line.startswith("please provide your answer"):
        return True
    if lower_line.startswith("please respond") and ("letter" in lower_line or "option" in lower_line):
        return True
    if lower_line.startswith("respond with only") and ("letter" in lower_line or "option" in lower_line):
        return True
    if lower_line.startswith("answer with") and ("letter" in lower_line or "option" in lower_line):
        return True
    if lower_line.startswith("answer the question using"):
        return True
    if lower_line.startswith("please only output") and ("answer" in lower_line or "option" in lower_line):
        return True
    if lower_line.startswith("do not generate any intermediate reasoning process"):
        return True
    if lower_line.startswith("please display your reasoning") and "final answer" in lower_line:
        return True
    if "stating the letter" in lower_line and "option" in lower_line:
        return True
    if "correct option" in lower_line and ("respond" in lower_line or "answer" in lower_line):
        return True
    return False
