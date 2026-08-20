"""Deterministic video decoding and frame sampling for AVIOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Frame-sampling settings used by training and inference."""

    fps: int = 2
    max_frames: int = 224
    force_uniform: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_frames < 0:
            raise ValueError("max_frames must be nonnegative")


@dataclass(frozen=True)
class VideoSample:
    """Decoded RGB frames and the physical-frame sampling record."""

    frames: np.ndarray
    frame_indices: tuple[int, ...]
    frame_times: tuple[float, ...]
    duration: float
    native_fps: float
    physical_frame_count: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def frame_time_text(self) -> str:
        return ",".join(f"{value:.2f}s" for value in self.frame_times)


def select_frame_indices(
    total_frames: int,
    native_fps: float,
    config: VideoSamplingConfig,
) -> tuple[list[int], list[float], str]:
    """Apply the training-time FPS stride followed by the frame cap."""

    total_frames = int(total_frames)
    native_fps = float(native_fps)
    if total_frames <= 0:
        raise RuntimeError("video contains no physical frames")
    if native_fps <= 0:
        native_fps = float(config.fps)

    stride = max(1, round(native_fps / config.fps))
    indices = list(range(0, total_frames, stride))
    mode = "native_fps_stride"
    if config.max_frames > 0 and (
        len(indices) > config.max_frames or config.force_uniform
    ):
        target_frames = min(config.max_frames, total_frames)
        indices = np.linspace(
            0,
            total_frames - 1,
            target_frames,
            dtype=int,
        ).tolist()
        mode = (
            "uniform_full_physical_sequence_min_cap"
            if config.force_uniform
            else "uniform_full_physical_sequence_cap"
        )
    times = [index / native_fps for index in indices]
    return indices, times, mode


def decode_video(
    path: str | Path,
    config: VideoSamplingConfig = VideoSamplingConfig(),
) -> VideoSample:
    """Decode a video with Decord and deterministic OpenCV/PyAV fallbacks."""

    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    errors: list[str] = []
    for backend, decoder in (
        ("decord", _decode_decord),
        ("opencv", _decode_opencv),
        ("pyav", _decode_pyav),
    ):
        try:
            return decoder(video_path, config, backend)
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"all video decoders failed for {video_path}: " + " | ".join(errors)
    )


def preprocess_video(
    sample: VideoSample,
    video_processor: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert sampled RGB frames to the vision tower's input tensor."""

    tensor = video_processor.preprocess(
        sample.frames,
        return_tensors="pt",
    )["pixel_values"]
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def _selection(
    total_frames: int,
    native_fps: float,
    config: VideoSamplingConfig,
) -> tuple[list[int], list[float], str, float]:
    if native_fps <= 0:
        native_fps = float(config.fps)
    indices, times, mode = select_frame_indices(
        total_frames,
        native_fps,
        config,
    )
    return indices, times, mode, native_fps


def _sample(
    frames: Sequence[np.ndarray] | np.ndarray,
    indices: list[int],
    times: list[float],
    total_frames: int,
    native_fps: float,
    backend: str,
    mode: str,
    metadata: dict[str, Any] | None = None,
) -> VideoSample:
    array = np.stack(frames, axis=0) if not isinstance(frames, np.ndarray) else frames
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError(f"expected RGB video [T,H,W,3], got {array.shape}")
    if array.shape[0] != len(indices):
        raise RuntimeError(
            f"decoded {array.shape[0]} frames for {len(indices)} sampled indices"
        )
    details = {
        "selection_mode": mode,
        "unique_selected_frame_count": len(set(indices)),
        **(metadata or {}),
    }
    return VideoSample(
        frames=array,
        frame_indices=tuple(int(index) for index in indices),
        frame_times=tuple(float(value) for value in times),
        duration=float(total_frames / native_fps),
        native_fps=float(native_fps),
        physical_frame_count=int(total_frames),
        backend=backend,
        metadata=details,
    )


def _decode_decord(
    path: Path,
    config: VideoSamplingConfig,
    backend: str,
) -> VideoSample:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total_frames = len(reader)
    native_fps = float(reader.get_avg_fps())
    indices, times, mode, native_fps = _selection(
        total_frames,
        native_fps,
        config,
    )
    frames = reader.get_batch(indices).asnumpy()
    reader.seek(0)
    return _sample(
        frames,
        indices,
        times,
        total_frames,
        native_fps,
        backend,
        mode,
    )


def _decode_opencv(
    path: Path,
    config: VideoSamplingConfig,
    backend: str,
) -> VideoSample:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        indices, times, mode, native_fps = _selection(
            total_frames,
            native_fps,
            config,
        )
        decoded: dict[int, np.ndarray] = {}
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        missing = [index for index in indices if index not in decoded]
        if missing:
            raise RuntimeError(
                f"OpenCV failed to decode {len(missing)} sampled frames"
            )
        return _sample(
            [decoded[index] for index in indices],
            indices,
            times,
            total_frames,
            native_fps,
            backend,
            mode,
        )
    finally:
        capture.release()


def _decode_pyav(
    path: Path,
    config: VideoSamplingConfig,
    backend: str,
) -> VideoSample:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded = [
            frame
            for packet in container.demux(stream)
            for frame in packet.decode()
        ]
        total_frames = len(decoded)
        average_rate = getattr(stream, "average_rate", None)
        native_fps = float(average_rate) if average_rate is not None else 0.0
        if native_fps <= 0 and decoded:
            last_time = getattr(decoded[-1], "time", None)
            if last_time and last_time > 0:
                native_fps = total_frames / float(last_time)
        indices, times, mode, native_fps = _selection(
            total_frames,
            native_fps,
            config,
        )
        frames = [decoded[index].to_ndarray(format="rgb24") for index in indices]
        return _sample(
            frames,
            indices,
            times,
            total_frames,
            native_fps,
            backend,
            mode,
        )
    finally:
        container.close()
