"""Configuration and model helpers for AVIOT training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class ModelSettings:
    checkpoint: str
    local_files_only: bool = False
    torch_dtype: str = "bfloat16"
    attention: str = "flash_attention_2"
    initialize_missing_aviot: bool = False


@dataclass
class DataSettings:
    annotations: str
    video_root: str
    fps: int = 2
    max_frames: int = 224
    force_uniform: bool = False
    seed: int = 42


@dataclass
class OptimizationSettings:
    output_dir: str
    max_steps: int = 10903
    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    save_total_limit: int = 10
    logging_steps: int = 1
    dataloader_num_workers: int = 4
    model_max_length: int = 32768
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = True
    deepspeed: str | None = None
    report_to: str = "none"
    run_name: str = "aviot"
    seed: int = 42
    dataloader_drop_last: bool = True


@dataclass
class TrainingConfig:
    model: ModelSettings
    data: DataSettings
    optimization: OptimizationSettings


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration field {name!r} must be a mapping")
    return value


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"training config not found: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _require_mapping(value, "root")
    model = ModelSettings(**_require_mapping(root.get("model"), "model"))
    data = DataSettings(**_require_mapping(root.get("data"), "data"))
    optimization = OptimizationSettings(
        **_require_mapping(root.get("optimization"), "optimization")
    )
    return TrainingConfig(model=model, data=data, optimization=optimization)


def synchronize_vocab_size(model) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        return
    model.config.vocab_size = int(embeddings.weight.shape[0])


__all__ = [
    "DataSettings",
    "ModelSettings",
    "OptimizationSettings",
    "TrainingConfig",
    "load_training_config",
    "synchronize_vocab_size",
]
