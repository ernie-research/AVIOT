"""Command-line supervised training for AVIOT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from aviot.data.supervised import VideoSupervisedCollator, VideoSupervisedDataset
from aviot.data.video import VideoSamplingConfig
from aviot.modeling.loader import load_config
from aviot.modeling.qwen import AVIOTQwenForCausalLM
from aviot.training.trainer import AVIOTTrainer
from aviot.training.utils import load_training_config, synchronize_vocab_size


def _torch_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported torch dtype: {name!r}")
    return mapping[value]


def load_training_model(config):
    model_config = load_config(
        config.model.checkpoint,
        overrides={
            "local_files_only": config.model.local_files_only,
            "tokenizer_model_max_length": config.optimization.model_max_length,
            "tokenizer_padding_side": "right",
            "aviot_ratio_policy": "random",
        },
        local_files_only=config.model.local_files_only,
    )
    model_config.use_cache = False
    model, loading_info = AVIOTQwenForCausalLM.from_pretrained(
        config.model.checkpoint,
        config=model_config,
        torch_dtype=_torch_dtype(config.model.torch_dtype),
        attn_implementation=config.model.attention,
        local_files_only=config.model.local_files_only,
        output_loading_info=True,
    )
    missing = list(loading_info.get("missing_keys", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    if config.model.initialize_missing_aviot:
        allowed_prefixes = (
            "model.aviot_compressor.",
            "model.aviot_query_projector.",
            "model.aviot_thw_position_encoder.",
        )
        missing = [key for key in missing if not key.startswith(allowed_prefixes)]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint is not compatible with the public AVIOT model: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}, "
            f"mismatched={mismatched[:12]}"
        )
    synchronize_vocab_size(model)
    return model


def build_training_arguments(config, *, max_steps: int | None = None) -> TrainingArguments:
    values = config.optimization
    return TrainingArguments(
        output_dir=values.output_dir,
        overwrite_output_dir=False,
        max_steps=values.max_steps if max_steps is None else int(max_steps),
        num_train_epochs=values.num_train_epochs,
        per_device_train_batch_size=values.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=values.gradient_accumulation_steps,
        learning_rate=values.learning_rate,
        weight_decay=values.weight_decay,
        warmup_ratio=values.warmup_ratio,
        lr_scheduler_type=values.lr_scheduler_type,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=values.save_steps,
        save_total_limit=values.save_total_limit,
        logging_steps=values.logging_steps,
        dataloader_num_workers=values.dataloader_num_workers,
        dataloader_drop_last=values.dataloader_drop_last,
        bf16=values.bf16,
        fp16=False,
        tf32=values.tf32,
        gradient_checkpointing=values.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": values.gradient_checkpointing_use_reentrant
        },
        deepspeed=values.deepspeed,
        report_to=[] if values.report_to == "none" else [values.report_to],
        run_name=values.run_name,
        seed=values.seed,
        data_seed=values.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )


def train(
    config_path: str | Path,
    *,
    resume_from_checkpoint: str | bool | None = None,
    max_steps: int | None = None,
) -> None:
    config = load_training_config(config_path)
    set_seed(config.optimization.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.checkpoint,
        use_fast=False,
        local_files_only=config.model.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = int(config.optimization.model_max_length)

    model = load_training_model(config)
    if config.optimization.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()
    tower = model.get_vision_tower()
    dataset = VideoSupervisedDataset(
        annotations=config.data.annotations,
        video_root=config.data.video_root,
        tokenizer=tokenizer,
        video_processor=tower.video_processor,
        sampling=VideoSamplingConfig(
            fps=config.data.fps,
            max_frames=config.data.max_frames,
            force_uniform=config.data.force_uniform,
        ),
        seed=config.data.seed,
    )
    collator = VideoSupervisedCollator(
        tokenizer=tokenizer,
        model_max_length=config.optimization.model_max_length,
    )
    trainer = AVIOTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=build_training_arguments(config, max_steps=max_steps),
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(config.optimization.output_dir)
    tokenizer.save_pretrained(config.optimization.output_dir)
    trainer.save_state()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="resume from the latest checkpoint or from the supplied path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the configured number of optimizer steps",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_args(argv)
    train(
        arguments.config,
        resume_from_checkpoint=arguments.resume_from_checkpoint,
        max_steps=arguments.max_steps,
    )


if __name__ == "__main__":
    main()
