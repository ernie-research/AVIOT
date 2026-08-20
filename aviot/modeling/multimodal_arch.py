"""AVIOT video encoding and multimodal-prefix construction.

The implementation intentionally contains only the video path used by the
released AVIOT model: SigLIP features, THW positional augmentation, AVIOT
compression, multimodal projection, 2-D pooling, and visual-prefix insertion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aviot.mm_utils import IGNORE_INDEX, VIDEO_TOKEN_INDEX
from .projector import build_multimodal_projector
from .temporal import AVIOTTemporalCompressor
from .vision import build_vision_tower


class AVIOTTHWPositionEncoder(nn.Module):
    """Add a learned Fourier encoding of normalized time, height, and width."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_bands: int = 16,
        temporal_scale: float = 1.0,
        spatial_scale: float = 0.5,
        warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_bands = int(num_bands)
        self.temporal_scale = float(temporal_scale)
        self.spatial_scale = float(spatial_scale)
        self.warmup_steps = int(warmup_steps)
        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self.proj = nn.Linear(6 * self.num_bands, self.embed_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def _warmup_scale(self, step: Optional[int]) -> float:
        if self.warmup_steps <= 0 or step is None:
            return 1.0
        return min(1.0, max(0.0, float(step) / float(self.warmup_steps)))

    def forward(
        self,
        features: torch.Tensor,
        *,
        step: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if features.ndim != 3:
            raise ValueError(f"expected video features [T,S,D], got {tuple(features.shape)}")
        scale = self._warmup_scale(step)
        frames, spatial_tokens, dim = features.shape
        if dim != self.embed_dim:
            raise ValueError(f"expected feature dimension {self.embed_dim}, got {dim}")
        side = math.isqrt(spatial_tokens)
        if side * side != spatial_tokens:
            raise ValueError(f"AVIOT requires a square spatial layout, got S={spatial_tokens}")
        if scale <= 0.0:
            return features, {
                "thw_position_scale": scale,
                "thw_grid_size": side,
                "thw_residual_rms": 0.0,
            }

        device = features.device
        coordinate_dtype = torch.float32
        time = torch.linspace(
            -1.0,
            1.0,
            frames,
            device=device,
            dtype=coordinate_dtype,
        ).view(frames, 1).expand(frames, spatial_tokens)
        positions = torch.arange(spatial_tokens, device=device)
        height = torch.div(positions, side, rounding_mode="floor").to(coordinate_dtype)
        width = (positions % side).to(coordinate_dtype)
        height = height.div(max(1, side - 1)).mul(2.0).sub(1.0)
        width = width.div(max(1, side - 1)).mul(2.0).sub(1.0)
        coordinates = torch.stack(
            (
                time * self.temporal_scale,
                height.view(1, -1).expand(frames, -1) * self.spatial_scale,
                width.view(1, -1).expand(frames, -1) * self.spatial_scale,
            ),
            dim=-1,
        )
        frequencies = (
            2.0 ** torch.arange(self.num_bands, device=device, dtype=coordinate_dtype)
        ) * math.pi
        phases = coordinates.unsqueeze(-1) * frequencies
        encoding = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(-2)
        residual = self.proj(
            encoding.flatten(0, 1).to(dtype=self.proj.weight.dtype)
        ).view(frames, spatial_tokens, dim)
        residual = residual.to(dtype=features.dtype)
        output = features + residual * scale
        return output, {
            "thw_position_scale": scale,
            "thw_grid_size": side,
            "thw_residual_rms": float(
                residual.detach().float().square().mean().sqrt().cpu().item()
            ),
        }


class AVIOTMetaModel:
    """Attach the final AVIOT visual modules to a Qwen2 decoder model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vision_tower = build_vision_tower(
            config,
            delay_load=bool(getattr(config, "delay_load_vision_tower", False)),
        )
        self.multimodal_projector = build_multimodal_projector(config)
        self.video_row_separator = nn.Parameter(
            torch.empty(config.hidden_size, dtype=self.dtype)
        )
        self._initialize_aviot(config)

    def _initialize_aviot(self, config) -> None:
        vision_dim = int(getattr(config, "vision_hidden_size", 1152))
        language_dim = int(config.hidden_size)
        self.aviot_query_projector = nn.Linear(language_dim, vision_dim, bias=False)
        self.aviot_thw_position_encoder = AVIOTTHWPositionEncoder(
            vision_dim,
            num_bands=int(getattr(config, "aviot_thw_num_bands", 16)),
            temporal_scale=float(getattr(config, "aviot_thw_temporal_scale", 1.0)),
            spatial_scale=float(getattr(config, "aviot_thw_spatial_scale", 0.5)),
            warmup_steps=int(getattr(config, "aviot_thw_warmup_steps", 500)),
        )
        self.aviot_compressor = AVIOTTemporalCompressor(
            embed_dim=vision_dim,
            k_video=int(getattr(config, "aviot_default_supports", 8)),
            cost_dim=int(getattr(config, "aviot_cost_dim", 256)),
            num_temporal_segments=int(getattr(config, "aviot_temporal_segments", 4)),
            min_budget_per_segment=1,
            eps=float(getattr(config, "aviot_entropy", 0.10)),
            rho_s=float(getattr(config, "aviot_source_relaxation", 0.5)),
            rho_t=float(getattr(config, "aviot_target_relaxation", 5.0)),
            sinkhorn_iter=int(getattr(config, "aviot_sinkhorn_iterations", 20)),
            num_ot_rounds=int(getattr(config, "aviot_global_refinement_rounds", 5)),
            lambda_ot=float(getattr(config, "aviot_transport_loss_weight", 1.0)),
            eta_mass=float(getattr(config, "aviot_mass_loss_weight", 0.01)),
            progressive_ratios=getattr(config, "aviot_progressive_ratios", (0.75, 0.5, 0.25)),
            progressive_round_to=int(getattr(config, "aviot_progressive_round_to", 8)),
            progressive_max_input_frames=int(getattr(config, "aviot_max_input_frames", 224)),
            final_ratio_choices=getattr(
                config,
                "aviot_training_ratios",
                tuple(2.0 + 0.5 * index for index in range(17)),
            ),
            final_ratio_policy=str(getattr(config, "aviot_ratio_policy", "random")),
            allocation_temperature_initial=float(
                getattr(config, "aviot_allocation_temperature_initial", 1.0)
            ),
            allocation_temperature_final=float(
                getattr(config, "aviot_allocation_temperature_final", 0.1)
            ),
            allocation_temperature_anneal_steps=int(
                getattr(config, "aviot_allocation_warmup_steps", 500)
            ),
            segment_prior_alpha=float(getattr(config, "aviot_question_allocation_weight", 0.3)),
            segment_prior_warmup_steps=int(getattr(config, "aviot_allocation_warmup_steps", 500)),
            multiscale_blocks=(27, 9, 3),
            multiscale_parent_prior=(
                float(getattr(config, "aviot_medium_parent_weight", 0.5)),
                float(getattr(config, "aviot_local_parent_weight", 0.25)),
            ),
            multiscale_eps=getattr(config, "aviot_multiscale_entropy", (0.10, 0.12, 0.15)),
            multiscale_rounds=getattr(config, "aviot_multiscale_refinement_rounds", (3, 2, 1)),
            multiscale_warmup_steps=int(getattr(config, "aviot_multiscale_warmup_steps", 500)),
            regional_transport_weight=float(getattr(config, "aviot_regional_transport_weight", 1.0)),
            continuity_weight=float(getattr(config, "aviot_continuity_weight", 0.01)),
            continuity_sigma=float(getattr(config, "aviot_continuity_scale", 0.20)),
            gate_hidden_dim=int(getattr(config, "aviot_gate_hidden_size", 32)),
            gate_global_floor=float(getattr(config, "aviot_gate_global_floor", 0.20)),
            gate_temperature=float(getattr(config, "aviot_gate_temperature", 0.05)),
            gate_tv_weight=float(getattr(config, "aviot_gate_tv_weight", 0.001)),
            gate_balance_weight=float(getattr(config, "aviot_gate_balance_weight", 0.5)),
            gate_balance_target=getattr(config, "aviot_gate_balance_target", (0.35, 0.45, 0.20)),
            gate_entropy_weight=float(getattr(config, "aviot_gate_entropy_weight", 0.5)),
            gate_entropy_floor=float(getattr(config, "aviot_gate_entropy_floor", 0.85)),
        )

    def get_vision_tower(self):
        return self.vision_tower


class AVIOTMetaForCausalLM(ABC):
    """Video-prefix behavior shared by the AVIOT causal language model."""

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _apply_aviot_aux_loss(self, outputs, *, labels=None):
        model = self.get_model()
        transport_loss = getattr(model, "_aviot_transport_loss", None)
        if labels is None or transport_loss is None:
            return outputs
        compressor = model.aviot_compressor
        auxiliary_loss = compressor.aviot_aux_loss(
            transport_loss,
            mass_distribution=getattr(model, "_aviot_mass_distribution", None),
        )
        query_anchor = auxiliary_loss.new_zeros(())
        for parameter in model.aviot_query_projector.parameters():
            if parameter.requires_grad:
                query_anchor = query_anchor + parameter.float().sum() * 0.0
        auxiliary_loss = auxiliary_loss + query_anchor
        if getattr(outputs, "loss", None) is not None:
            language_loss = outputs.loss.detach().float()
            outputs.loss = outputs.loss + auxiliary_loss.to(outputs.loss)
            compressor.last_aux_stats["lm_loss"] = float(language_loss.item())
            compressor.last_aux_stats["total_loss"] = float(outputs.loss.detach().float().item())
            return outputs
        if isinstance(outputs, tuple) and outputs and outputs[0] is not None:
            total = outputs[0] + auxiliary_loss.to(outputs[0])
            return (total,) + outputs[1:]
        return outputs

    @staticmethod
    def _question_range_mask(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_token_ranges,
    ) -> Optional[torch.Tensor]:
        if question_token_ranges is None:
            return None
        ranges = torch.as_tensor(
            question_token_ranges,
            device=input_ids.device,
            dtype=torch.long,
        )
        if ranges.ndim == 2 and ranges.shape[-1] == 2:
            ranges = ranges.unsqueeze(1)
        if ranges.ndim != 3 or ranges.shape[-1] != 2:
            raise ValueError("question_token_ranges must have shape [B,R,2]")
        batch_size, sequence_length = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_index in range(min(batch_size, ranges.shape[0])):
            for start, end in ranges[batch_index].tolist():
                if start < 0 or end <= start:
                    continue
                start = min(sequence_length, max(0, int(start)))
                end = min(sequence_length, max(0, int(end)))
                mask[batch_index, start:end] = True
        mask &= attention_mask.bool()
        mask &= input_ids.ne(VIDEO_TOKEN_INDEX)
        return mask if bool(mask.any()) else None

    @staticmethod
    def _fallback_question_mask(
        sample_input_ids: torch.Tensor,
        sample_prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = sample_prompt_mask.clone()
        visual_positions = torch.where(sample_input_ids.eq(VIDEO_TOKEN_INDEX))[0]
        if visual_positions.numel():
            mask[: int(visual_positions[-1].item()) + 1] = False
        candidates = torch.where(mask)[0]
        if candidates.numel() > 2:
            newlines = torch.where(sample_input_ids[candidates].eq(198))[0]
            if newlines.numel() and int(newlines[-1].item()) + 1 < candidates.numel():
                start = int(candidates[int(newlines[-1].item()) + 1].item())
                refined = torch.zeros_like(mask)
                refined[start:] = mask[start:]
                if bool(refined.any()):
                    return refined
        return mask if bool(mask.any()) else sample_prompt_mask

    def _pool_question_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        question_token_ranges,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        model = self.get_model()
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        safe_ids = input_ids.masked_fill(input_ids.eq(VIDEO_TOKEN_INDEX), 0)
        prompt = valid & input_ids.ne(VIDEO_TOKEN_INDEX)
        if labels is not None:
            prompt &= labels.eq(IGNORE_INDEX)
        explicit = self._question_range_mask(input_ids, valid, question_token_ranges)
        masks = []
        for index in range(input_ids.shape[0]):
            if explicit is not None and bool(explicit[index].any()):
                candidate = explicit[index] & prompt[index]
                masks.append(
                    candidate
                    if bool(candidate.any())
                    else self._fallback_question_mask(input_ids[index], prompt[index])
                )
            else:
                masks.append(self._fallback_question_mask(input_ids[index], prompt[index]))
        question_mask = torch.stack(masks)
        embeddings = model.embed_tokens(safe_ids)
        visual_queries: list[torch.Tensor] = []
        language_queries: list[torch.Tensor] = []
        for sample_embeddings, sample_mask in zip(embeddings, question_mask):
            selected = sample_embeddings[sample_mask]
            if selected.numel() == 0:
                raise ValueError("a question query could not be formed from the prompt")
            pooled = selected.mean(dim=0).detach()
            language_queries.append(pooled)
            weight = model.aviot_query_projector.weight
            projected = model.aviot_query_projector(
                pooled.to(device=weight.device, dtype=weight.dtype)
            )
            visual_queries.append(projected.to(device=embeddings.device))
        return visual_queries, language_queries

    def _encode_vision_frames(self, frames: torch.Tensor) -> torch.Tensor:
        chunk_size = int(getattr(self.config, "vision_tower_batch_size", 0) or 0)
        tower = self.get_vision_tower()
        if chunk_size <= 0 or frames.shape[0] <= chunk_size:
            return tower(frames)
        return torch.cat(
            [tower(frames[start : start + chunk_size]) for start in range(0, frames.shape[0], chunk_size)],
            dim=0,
        )

    def _pool_spatial_grid(self, features: torch.Tensor) -> torch.Tensor:
        stride = int(getattr(self.config, "spatial_pool_stride", 2))
        mode = str(getattr(self.config, "spatial_pool_mode", "bilinear"))
        if stride != 2 or mode != "bilinear":
            raise ValueError("the released AVIOT model requires stride-2 bilinear spatial pooling")
        supports, tokens, dim = features.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError(f"expected a square feature grid, got {tokens} tokens")
        grid = features.view(supports, side, side, dim).permute(0, 3, 1, 2)
        output_side = math.ceil(side / stride)
        grid = F.interpolate(grid, size=(output_side, output_side), mode="bilinear")
        return grid.permute(0, 2, 3, 1).reshape(supports, output_side * output_side, dim)

    def _format_visual_prefix(self, features: torch.Tensor) -> torch.Tensor:
        supports, tokens, dim = features.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError(f"expected a square pooled grid, got {tokens} tokens")
        grid = features.view(supports, side, side, dim)
        separator = self.get_model().video_row_separator.to(features).view(1, 1, 1, dim)
        separator = separator.expand(supports, side, 1, dim)
        return torch.cat((grid, separator), dim=2).reshape(
            supports * side * (side + 1), dim
        )

    def _frame_relevance(
        self,
        features: torch.Tensor,
        language_query: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.get_model().multimodal_projector(features)
        frame_descriptors = F.normalize(projected.float().mean(dim=1), dim=-1)
        query = F.normalize(language_query.to(frame_descriptors).float(), dim=-1)
        return (frame_descriptors @ query).detach()

    def _encode_videos(
        self,
        videos: Sequence[torch.Tensor],
        visual_queries: Sequence[torch.Tensor],
        language_queries: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not videos:
            raise ValueError("at least one video is required")
        lengths = [int(video.shape[0]) for video in videos]
        if any(video.ndim != 4 for video in videos):
            raise ValueError("each video must have shape [T,C,H,W]")
        concatenated = torch.cat(list(videos), dim=0)
        encoded = self._encode_vision_frames(concatenated)
        raw_videos = torch.split(encoded, lengths, dim=0)

        model = self.get_model()
        model._aviot_transport_loss = encoded.new_zeros((), dtype=torch.float32)
        model._aviot_stats = []
        mass_distributions = []
        prefixes = []
        compressor = model.aviot_compressor
        for raw_features, visual_query, language_query in zip(
            raw_videos,
            visual_queries,
            language_queries,
        ):
            motion_features = raw_features.detach()
            step = int(compressor.current_step.item())
            positioned, position_stats = model.aviot_thw_position_encoder(raw_features, step=step)
            relevance = self._frame_relevance(positioned, language_query)
            compressed, transport_loss, stats = compressor.compress_video(
                positioned,
                query=visual_query.to(positioned.device),
                query_source="question",
                frame_relevance=relevance,
                motion_features=motion_features,
                extra_stats=position_stats,
            )
            model._aviot_transport_loss = model._aviot_transport_loss + transport_loss
            model._aviot_stats.append(stats)
            if compressor.mass_distribution is not None:
                mass_distributions.append(compressor.mass_distribution.detach())
            projected = model.multimodal_projector(compressed)
            prefixes.append(self._format_visual_prefix(self._pool_spatial_grid(projected)))
        model._aviot_mass_distribution = (
            torch.cat(mass_distributions, dim=0) if mass_distributions else None
        )
        if model._aviot_mass_distribution is not None:
            compressor._mass_distribution = model._aviot_mass_distribution
        return prefixes

    @staticmethod
    def _normalize_videos(videos, batch_size: int) -> list[torch.Tensor]:
        if torch.is_tensor(videos):
            if videos.ndim != 5:
                raise ValueError("videos tensor must have shape [B,T,C,H,W]")
            result = list(videos.unbind(0))
        else:
            result = list(videos)
        if len(result) != batch_size:
            raise ValueError(f"received {len(result)} videos for a batch of {batch_size}")
        return result

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        labels: Optional[torch.Tensor],
        videos,
        question_token_ranges=None,
    ):
        if videos is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        batch_size = input_ids.shape[0]
        video_list = self._normalize_videos(videos, batch_size)
        visual_queries, language_queries = self._pool_question_queries(
            input_ids,
            attention_mask,
            labels,
            question_token_ranges,
        )
        visual_prefixes = self._encode_videos(video_list, visual_queries, language_queries)

        original_attention_mask = attention_mask
        original_position_ids = position_ids
        active_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        working_labels = (
            torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels
        )
        sample_ids = [row[mask] for row, mask in zip(input_ids, active_mask)]
        sample_labels = [row[mask] for row, mask in zip(working_labels, active_mask)]

        embedded_samples = []
        output_labels = []
        embedding_layer = self.get_model().embed_tokens
        for ids, label_row, prefix in zip(sample_ids, sample_labels, visual_prefixes):
            positions = torch.where(ids.eq(VIDEO_TOKEN_INDEX))[0]
            if positions.numel() != 1:
                raise ValueError(
                    "AVIOT requires exactly one video placeholder per training or inference sample"
                )
            position = int(positions[0].item())
            before = embedding_layer(ids[:position])
            after = embedding_layer(ids[position + 1 :])
            embedded_samples.append(torch.cat((before, prefix.to(before), after), dim=0))
            visual_labels = torch.full(
                (prefix.shape[0],),
                IGNORE_INDEX,
                dtype=label_row.dtype,
                device=label_row.device,
            )
            output_labels.append(
                torch.cat((label_row[:position], visual_labels, label_row[position + 1 :]), dim=0)
            )

        maximum_length = int(getattr(self.config, "tokenizer_model_max_length", 32768))
        lengths = [int(item.shape[0]) for item in embedded_samples]
        if max(lengths) > maximum_length:
            raise RuntimeError(
                "AVIOT multimodal sequence exceeds tokenizer_model_max_length: "
                f"max={max(lengths)}, limit={maximum_length}, lengths={lengths}"
            )
        padded_length = max(lengths)
        dim = embedded_samples[0].shape[-1]
        padded_embeddings = embedded_samples[0].new_zeros((batch_size, padded_length, dim))
        padded_labels = output_labels[0].new_full(
            (batch_size, padded_length),
            IGNORE_INDEX,
        )
        padded_attention = active_mask.new_zeros((batch_size, padded_length))
        padded_positions = input_ids.new_zeros((batch_size, padded_length))
        padding_side = str(getattr(self.config, "tokenizer_padding_side", "right"))
        for index, (embeddings, label_row) in enumerate(zip(embedded_samples, output_labels)):
            length = embeddings.shape[0]
            destination = slice(padded_length - length, padded_length) if padding_side == "left" else slice(0, length)
            padded_embeddings[index, destination] = embeddings
            padded_labels[index, destination] = label_row
            padded_attention[index, destination] = True
            padded_positions[index, destination] = torch.arange(length, device=input_ids.device)

        return (
            None,
            None if original_position_ids is None else padded_positions,
            None if original_attention_mask is None else padded_attention.to(original_attention_mask.dtype),
            past_key_values,
            padded_embeddings,
            None if labels is None else padded_labels,
        )
