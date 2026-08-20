"""Thin Transformers Trainer integration for AVIOT."""

from __future__ import annotations

from typing import Any

from transformers import Trainer, TrainerCallback


def get_aviot_compressor(model):
    """Return the compressor through common distributed wrappers."""

    current = model
    while hasattr(current, "module"):
        current = current.module
    if hasattr(current, "get_model"):
        current = current.get_model()
    return getattr(current, "aviot_compressor", None)


class AVIOTStepCallback(TrainerCallback):
    """Keep schedule buffers synchronized with optimizer steps and resumes."""

    @staticmethod
    def _sync(state, model) -> None:
        compressor = get_aviot_compressor(model)
        if compressor is None:
            return
        compressor.current_step.fill_(int(state.global_step))
        compressor.update_allocation_temperature()

    def on_train_begin(self, args, state, _control, model=None, **kwargs):
        if model is not None:
            self._sync(state, model)

    def on_step_end(self, args, state, _control, model=None, **kwargs):
        if model is not None:
            self._sync(state, model)


class AVIOTTrainer(Trainer):
    """Trainer with AVIOT schedule synchronization and compact loss logging."""

    def __init__(self, *args, **kwargs) -> None:
        callbacks = list(kwargs.pop("callbacks", ()) or ())
        callbacks.append(AVIOTStepCallback())
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        compressor = get_aviot_compressor(self.model)
        if compressor is not None:
            auxiliary = getattr(compressor, "last_aux_stats", {}) or {}
            for source, destination in (
                ("lm_loss", "loss/language"),
                ("transport_loss", "loss/transport"),
                ("aux_loss", "loss/aviot"),
                ("total_loss", "loss/total"),
            ):
                value = auxiliary.get(source)
                if value is not None:
                    logs.setdefault(destination, float(value))
        super().log(logs, *args, **kwargs)


__all__ = ["AVIOTStepCallback", "AVIOTTrainer", "get_aviot_compressor"]
