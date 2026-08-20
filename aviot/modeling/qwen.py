"""Qwen2 language model with the AVIOT multimodal input path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from aviot.mm_utils import IGNORE_INDEX
from .multimodal_arch import AVIOTMetaForCausalLM, AVIOTMetaModel


class AVIOTQwenConfig(Qwen2Config):
    model_type = "aviot_qwen"

    def __init__(self, *args, **kwargs):
        text_config = kwargs.get("text_config")
        if "vocab_size" not in kwargs and isinstance(text_config, dict):
            if text_config.get("vocab_size") is not None:
                kwargs["vocab_size"] = text_config["vocab_size"]
        defaults = {
            "vision_tower": None,
            "delay_load_vision_tower": False,
            "local_files_only": False,
            "vision_select_layer": -1,
            "vision_hidden_size": 1152,
            "aviot_vision_weights_embedded": False,
            "multimodal_projector_type": "mlp2x_gelu",
            "spatial_pool_mode": "bilinear",
            "spatial_pool_stride": 2,
            "vision_tower_batch_size": 0,
            "tokenizer_model_max_length": 32768,
            "tokenizer_padding_side": "right",
            "aviot_default_supports": 8,
            "aviot_cost_dim": 256,
            "aviot_temporal_segments": 4,
            "aviot_entropy": 0.10,
            "aviot_source_relaxation": 0.5,
            "aviot_target_relaxation": 5.0,
            "aviot_sinkhorn_iterations": 20,
            "aviot_global_refinement_rounds": 5,
            "aviot_transport_loss_weight": 1.0,
            "aviot_mass_loss_weight": 0.01,
            "aviot_progressive_ratios": [0.75, 0.50, 0.25],
            "aviot_progressive_round_to": 8,
            "aviot_max_input_frames": 224,
            "aviot_training_ratios": [2.0 + 0.5 * index for index in range(17)],
            "aviot_ratio_policy": "random",
            "aviot_thw_num_bands": 16,
            "aviot_thw_temporal_scale": 1.0,
            "aviot_thw_spatial_scale": 0.5,
            "aviot_thw_warmup_steps": 500,
            "aviot_allocation_temperature_initial": 1.0,
            "aviot_allocation_temperature_final": 0.1,
            "aviot_allocation_warmup_steps": 500,
            "aviot_question_allocation_weight": 0.3,
            "aviot_region_position_bands": 8,
            "aviot_medium_parent_weight": 0.50,
            "aviot_local_parent_weight": 0.25,
            "aviot_multiscale_entropy": [0.10, 0.12, 0.15],
            "aviot_multiscale_refinement_rounds": [3, 2, 1],
            "aviot_multiscale_warmup_steps": 500,
            "aviot_regional_transport_weight": 1.0,
            "aviot_continuity_weight": 0.01,
            "aviot_continuity_scale": 0.20,
            "aviot_gate_hidden_size": 32,
            "aviot_gate_global_floor": 0.20,
            "aviot_gate_temperature": 0.05,
            "aviot_gate_tv_weight": 0.001,
            "aviot_gate_balance_weight": 0.5,
            "aviot_gate_balance_target": [0.35, 0.45, 0.20],
            "aviot_gate_entropy_weight": 0.5,
            "aviot_gate_entropy_floor": 0.85,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        super().__init__(*args, **kwargs)

    def get_text_config(self, decoder=None, encoder=None):
        return self

    @property
    def video_token_index(self) -> int:
        from aviot.mm_utils import VIDEO_TOKEN_INDEX

        return VIDEO_TOKEN_INDEX


class AVIOTQwenModel(AVIOTMetaModel, Qwen2Model):
    config_class = AVIOTQwenConfig

    def __init__(self, config: AVIOTQwenConfig):
        super().__init__(config)


class AVIOTQwenForCausalLM(Qwen2ForCausalLM, AVIOTMetaForCausalLM):
    config_class = AVIOTQwenConfig

    def __init__(self, config: AVIOTQwenConfig):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = AVIOTQwenConfig.model_type
        config.rope_scaling = None
        self.model = AVIOTQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights(self.lm_head)

    def get_model(self):
        return self.model

    def _supervised_positions_loss(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
        shifted_labels = labels[..., 1:].contiguous()
        mask = shifted_labels.ne(IGNORE_INDEX)
        if not bool(mask.any().item()):
            return None, None
        selected_hidden = hidden_states[..., :-1, :][mask]
        logits = self.lm_head(selected_hidden).float()
        return logits, shifted_labels[mask].to(logits.device)

    def _forward_training_loss(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits, target = self._supervised_positions_loss(outputs[0], labels)
        if logits is None:
            return None
        loss = CrossEntropyLoss()(logits, target)
        if not return_dict:
            return (loss, logits) + outputs[1:]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        videos=None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        question_token_ranges: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                videos,
                question_token_ranges=question_token_ranges,
            )
        outputs = None
        if self.training and labels is not None:
            outputs = self._forward_training_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if outputs is None:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        return self._apply_aviot_aux_loss(outputs, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        videos=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        question_token_ranges = kwargs.pop("question_token_ranges", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is not supported by multimodal generate")
        if videos is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                videos,
                question_token_ranges=question_token_ranges,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        videos = kwargs.pop("videos", None)
        prepared = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if videos is not None:
            prepared["videos"] = videos
        return prepared


AutoConfig.register(AVIOTQwenConfig.model_type, AVIOTQwenConfig)
AutoModelForCausalLM.register(AVIOTQwenConfig, AVIOTQwenForCausalLM)
