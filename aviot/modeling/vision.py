"""SigLIP visual encoder construction for AVIOT."""

from __future__ import annotations

from pathlib import Path

from .siglip import SigLIPVideoTower


def _resolve_vision_tower_path(config, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    checkpoint = Path(str(getattr(config, "_name_or_path", ""))).expanduser()
    if checkpoint.is_dir():
        bundled = checkpoint / path
        if bundled.is_dir():
            return str(bundled.resolve())
    return str(path)


def build_vision_tower(config, *, delay_load: bool = False) -> SigLIPVideoTower:
    path = getattr(config, "vision_tower", None)
    if not path:
        raise ValueError("config.vision_tower must identify SigLIP SO400M patch14/384")
    return SigLIPVideoTower(
        _resolve_vision_tower_path(config, str(path)),
        delay_load=delay_load,
        local_files_only=bool(getattr(config, "local_files_only", False)),
        select_layer=int(getattr(config, "vision_select_layer", -1)),
        weights_embedded=bool(
            getattr(config, "aviot_vision_weights_embedded", False)
        ),
    )
