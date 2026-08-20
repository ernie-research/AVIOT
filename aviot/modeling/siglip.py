"""SigLIP frame encoder used by the public AVIOT model.

The released model uses the vision branch of SigLIP SO400M at 384 pixels.
This module keeps the dependency boundary small: Transformers owns the
checkpoint and encoder implementation, while AVIOT owns frame batching and
the video-specific output contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import SiglipConfig, SiglipVisionModel
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import ChannelDimension, to_numpy_array

try:
    from transformers.image_utils import PILImageResampling
except ImportError:  # Transformers 4.56 exposes the same values through Pillow.
    from PIL import Image

    PILImageResampling = Image.Resampling


class SigLIPFrameProcessor:
    """Convert RGB video frames to the SigLIP input tensor."""

    def __init__(self, *, size: int = 384) -> None:
        self.size = int(size)

    @classmethod
    def from_pretrained(cls, path: str | Path, *, local_files_only: bool = False):
        del path, local_files_only
        return cls()

    def preprocess(self, frames: Any, *, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("SigLIPFrameProcessor supports return_tensors='pt' only")
        if isinstance(frames, (np.ndarray, torch.Tensor)):
            if frames.ndim != 4:
                raise ValueError(
                    f"expected a frame batch with four dimensions, got {tuple(frames.shape)}"
                )
            frame_list = list(frames)
        elif isinstance(frames, (list, tuple)):
            frame_list = list(frames)
        else:
            frame_list = [frames]
        if not frame_list:
            raise ValueError("at least one RGB frame is required")

        processed: list[np.ndarray] = []
        for frame in frame_list:
            image = to_numpy_array(convert_to_rgb(frame))
            image = resize(
                image,
                size=(self.size, self.size),
                resample=PILImageResampling.BICUBIC,
                data_format=ChannelDimension.FIRST,
            )
            image = rescale(
                image,
                scale=1.0 / 255.0,
                data_format=ChannelDimension.FIRST,
            )
            image = normalize(
                image,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                data_format=ChannelDimension.FIRST,
            )
            image = to_channel_dimension_format(
                image,
                ChannelDimension.FIRST,
                input_channel_dim=ChannelDimension.FIRST,
            )
            processed.append(image)

        return {"pixel_values": torch.from_numpy(np.stack(processed))}


class SigLIPVideoTower(nn.Module):
    """Batched SigLIP patch features with a video-oriented public API."""

    def __init__(
        self,
        checkpoint: str,
        *,
        delay_load: bool = False,
        local_files_only: bool = False,
        select_layer: int = -1,
        weights_embedded: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint = str(checkpoint)
        self.local_files_only = bool(local_files_only)
        self.weights_embedded = bool(weights_embedded)
        if int(select_layer) != -1:
            raise ValueError("the released AVIOT vision path requires select_layer=-1")
        # The final released checkpoint removes the last encoder block and
        # bypasses the pooled head.  ``-1`` therefore denotes the exact patch
        # features consumed by the original final training path.
        self.select_layer = -1
        self._loaded = False
        self.encoder: SiglipVisionModel | None = None
        self.video_processor = SigLIPFrameProcessor.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.hidden_size = 1152
        self.num_patches = 729
        self.grid_size = 27
        if not delay_load:
            self.load_model()

    def load_model(self) -> None:
        if self._loaded:
            return
        if self.weights_embedded:
            siglip_config = SiglipConfig.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            vision_config = siglip_config.vision_config
            vision_config._attn_implementation = "eager"
            self.encoder = SiglipVisionModel(vision_config)
        else:
            self.encoder = SiglipVisionModel.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
                attn_implementation="eager",
            )
        vision_model = getattr(self.encoder, "vision_model", None)
        if vision_model is None:
            raise RuntimeError("the SigLIP checkpoint has no vision_model branch")
        layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
        if layers is not None and len(layers) > 0:
            del layers[-1:]
        if hasattr(vision_model, "head"):
            vision_model.head = nn.Identity()
        self.encoder.requires_grad_(False)
        self.hidden_size = int(self.encoder.config.hidden_size)
        image_size = int(self.encoder.config.image_size)
        patch_size = int(self.encoder.config.patch_size)
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self._loaded = True

    @property
    def device(self) -> torch.device:
        if self.encoder is None:
            return torch.device("cpu")
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        if self.encoder is None:
            return torch.float32
        return next(self.encoder.parameters()).dtype

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.load_model()
        if frames.ndim != 4:
            raise ValueError(f"expected frame tensor [T,C,H,W], got {tuple(frames.shape)}")
        assert self.encoder is not None
        input_dtype = frames.dtype
        pixel_values = frames.to(device=self.device, dtype=self.dtype)

        # The original final training path consumes the output of the last
        # retained encoder block, before SigLIP's post-layernorm.  Calling the
        # high-level vision model would expose the post-layernorm tensor as
        # ``last_hidden_state`` in current Transformers releases.  Use the
        # underlying modules explicitly so this boundary remains stable across
        # output-structure changes.
        vision_model = self.encoder.vision_model
        embeddings = vision_model.embeddings(pixel_values)
        encoder_output = vision_model.encoder(
            inputs_embeds=embeddings,
            output_hidden_states=False,
            return_dict=True,
        )
        features = encoder_output.last_hidden_state
        # SigLIP vision outputs only patch tokens for the checkpoint used by
        # AVIOT.  Keep the assertion explicit so a mismatched checkpoint fails
        # at the boundary instead of silently corrupting spatial routing.
        if features.shape[1] != self.num_patches:
            raise ValueError(
                f"expected {self.num_patches} patch tokens, got {features.shape[1]}"
            )
        return features.to(dtype=input_dtype)


__all__ = ["SigLIPFrameProcessor", "SigLIPVideoTower"]
