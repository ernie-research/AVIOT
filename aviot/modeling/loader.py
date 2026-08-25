"""Model and tokenizer loading for AVIOT checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .qwen import AVIOTQwenConfig, AVIOTQwenForCausalLM


_REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "vision_tower/config.json",
    "vision_tower/preprocessor_config.json",
)


def resolve_checkpoint(
    checkpoint: str | Path,
    *,
    local_files_only: bool = False,
) -> str:
    """Resolve a local AVIOT directory or download a Hugging Face snapshot."""
    raw_checkpoint = str(checkpoint)
    local_path = Path(raw_checkpoint).expanduser()
    if local_path.is_dir():
        resolved = local_path.resolve()
    else:
        if local_path.exists():
            raise NotADirectoryError(
                f"AVIOT checkpoint must be a directory: {local_path}"
            )
        if (
            local_path.is_absolute()
            or raw_checkpoint.startswith(("./", "../", "~"))
        ):
            raise FileNotFoundError(
                f"AVIOT checkpoint directory does not exist: {local_path}"
            )
        resolved = Path(
            snapshot_download(
                repo_id=raw_checkpoint,
                repo_type="model",
                local_files_only=local_files_only,
            )
        ).resolve()

    missing = [
        relative_path
        for relative_path in _REQUIRED_CHECKPOINT_FILES
        if not (resolved / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"incomplete AVIOT checkpoint at {resolved}: missing {missing}"
        )
    return str(resolved)


def load_config(
    checkpoint: str,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    local_files_only: bool = False,
) -> AVIOTQwenConfig:
    """Load a public AVIOT config without mutating the checkpoint."""
    config = AVIOTQwenConfig.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    if config.model_type != AVIOTQwenConfig.model_type:
        raise ValueError(
            f"expected an {AVIOTQwenConfig.model_type!r} checkpoint, "
            f"but config.model_type is {config.model_type!r}"
    )
    vision_tower = getattr(config, "vision_tower", None)
    if vision_tower:
        component_path = Path(str(vision_tower)).expanduser()
        checkpoint_path = Path(checkpoint).expanduser()
        if not component_path.is_absolute() and checkpoint_path.is_dir():
            bundled_path = checkpoint_path / component_path
            if bundled_path.is_dir():
                config.vision_tower = str(bundled_path.resolve())
    for key, value in (overrides or {}).items():
        setattr(config, key, value)
    return config


def load_pretrained_model(
    checkpoint: str | Path,
    *,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    config_overrides: Optional[Mapping[str, Any]] = None,
    local_files_only: bool = False,
):
    """Load the final AVIOT Qwen2 model and its tokenizer."""
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(torch_dtype).lower())
    if dtype is None:
        raise ValueError(f"unsupported torch_dtype={torch_dtype!r}")
    checkpoint = resolve_checkpoint(
        checkpoint,
        local_files_only=local_files_only,
    )
    config = load_config(
        checkpoint,
        overrides=config_overrides,
        local_files_only=True,
    )
    config.local_files_only = True
    tokenizer_source = tokenizer_path or checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=False,
        local_files_only=True if tokenizer_path is None else local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    config.tokenizer_padding_side = tokenizer.padding_side
    config.tokenizer_model_max_length = int(getattr(tokenizer, "model_max_length", 32768))
    config.use_cache = True
    device_map: Any = "auto" if device == "auto" else {"": device}
    model, loading_info = AVIOTQwenForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        local_files_only=True,
        output_loading_info=True,
    )
    failures = {
        name: values
        for name, values in loading_info.items()
        if values
    }
    if failures:
        summary = "; ".join(
            f"{name}={values[:8]!r}" for name, values in failures.items()
        )
        raise RuntimeError(f"AVIOT checkpoint did not load exactly: {summary}")
    model.eval()
    return tokenizer, model


def set_inference_ratio(model: AVIOTQwenForCausalLM, ratio: float) -> None:
    """Select one target ratio without changing learned parameters."""
    ratio = float(ratio)
    if ratio <= 1.0:
        raise ValueError("compression ratio must be greater than one")
    compressor = model.get_model().aviot_compressor
    if compressor is None:
        raise ValueError("the loaded model does not contain AVIOT")
    compressor.final_ratio_policy = "fixed_ratio"
    compressor.final_ratio_choices = [ratio]
    model.config.aviot_ratio_policy = "fixed_ratio"
    model.config.aviot_training_ratios = [ratio]
