"""Multimodal projector used by the AVIOT Qwen2 model."""

from __future__ import annotations

import torch.nn as nn


def build_multimodal_projector(config) -> nn.Module:
    """Build the fixed two-layer GELU projector used by the released model."""

    projector_type = getattr(config, "multimodal_projector_type", "mlp2x_gelu")
    if projector_type != "mlp2x_gelu":
        raise ValueError(
            "AVIOT supports the final 'mlp2x_gelu' multimodal projector only; "
            f"got {projector_type!r}"
        )
    vision_dim = int(getattr(config, "vision_hidden_size", 1152))
    language_dim = int(config.hidden_size)
    return nn.Sequential(
        nn.Linear(vision_dim, language_dim),
        nn.GELU(),
        nn.Linear(language_dim, language_dim),
    )
