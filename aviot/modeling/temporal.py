"""Temporal OT construction used by the released AVIOT model."""

from __future__ import annotations

import math
import os
import random
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale import AVIOTMultiscaleStage
from .regions import compute_feature_delta_motion


def _distributed_rank() -> int:
    """Read the launcher rank without requiring a distributed package."""
    for name in ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


class QueryModulatedMetric(nn.Module):
    """Learned question-conditioned metric used by all OT branches."""

    def __init__(self, embed_dim: int, cost_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cost_dim = int(cost_dim)
        self.query_dim = int(embed_dim)
        self.phi = nn.Sequential(
            nn.Linear(self.embed_dim, self.cost_dim),
            nn.GELU(),
            nn.Linear(self.cost_dim, self.cost_dim),
            nn.LayerNorm(self.cost_dim),
        )
        hidden = max(1, self.cost_dim // 2)
        self.h_theta = nn.Sequential(
            nn.Linear(self.query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.cost_dim),
        )

    def compute_cost_matrix(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        query: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source_space = self.phi(source)
        target_space = self.phi(target)
        if query is not None:
            query = query.reshape(-1, query.shape[-1])
            weights = F.softplus(self.h_theta(query.to(source_space.dtype)))
            source_space = source_space * weights
            target_space = target_space * weights
        source_norm = source_space.square().sum(dim=-1, keepdim=True)
        target_norm = target_space.square().sum(dim=-1, keepdim=True).transpose(0, 1)
        return (source_norm + target_norm - 2.0 * source_space @ target_space.transpose(0, 1)).clamp_min(0.0)


def unbalanced_sinkhorn(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    eps: float,
    rho_s: float,
    rho_t: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-step log-domain unbalanced Sinkhorn updates."""

    epsilon = max(float(eps), 1e-6)
    with torch.no_grad():
        maximum = cost.detach().amax()
    scale = torch.where(
        maximum > epsilon * 100.0,
        epsilon * 100.0 / (maximum + 1e-8),
        torch.ones_like(maximum),
    )
    cost = cost * scale
    log_source = source_weights.clamp_min(1e-20).log()
    log_target = target_weights.clamp_min(1e-20).log()
    f = torch.zeros_like(source_weights)
    g = torch.zeros_like(target_weights)
    source_rate = float(rho_s) / (float(rho_s) + epsilon)
    target_rate = float(rho_t) / (float(rho_t) + epsilon)
    for _ in range(max(1, int(iterations))):
        row_lse = torch.logsumexp((g.unsqueeze(0) - cost) / epsilon, dim=1)
        f = source_rate * (epsilon * log_source - epsilon * row_lse) + (1.0 - source_rate) * f
        column_lse = torch.logsumexp((f.unsqueeze(1) - cost) / epsilon, dim=0)
        g = target_rate * (epsilon * log_target - epsilon * column_lse) + (1.0 - target_rate) * g
    log_plan = (f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon
    return log_plan.exp(), log_plan


def barycentric_update(
    plan: torch.Tensor,
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = plan.sum(dim=0)
    supports = plan.transpose(0, 1) @ source / (mass.unsqueeze(-1) + 1e-8)
    return supports, mass


def _uniform_supports(source: torch.Tensor, count: int) -> torch.Tensor:
    indices = torch.linspace(
        0,
        source.shape[0] - 1,
        int(count),
        device=source.device,
        dtype=torch.float32,
    ).round().long()
    return source.index_select(0, indices)


def ot_consolidate_segment(
    source: torch.Tensor,
    budget: int,
    metric: QueryModulatedMetric,
    *,
    query: Optional[torch.Tensor],
    eps: float,
    rho_s: float,
    rho_t: float,
    sinkhorn_iterations: int,
    refinement_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct target supports and a source-to-target plan for one segment."""

    frames = int(source.shape[0])
    budget = max(1, min(int(budget), frames))
    if frames == budget:
        identity = torch.eye(frames, device=source.device, dtype=source.dtype)
        mass = torch.ones(frames, device=source.device, dtype=source.dtype)
        mass = mass / float(frames)
        return source, mass, source.new_zeros((), dtype=torch.float32), identity

    supports = _uniform_supports(source, budget)
    source_weights = torch.full(
        (frames,), 1.0 / frames, device=source.device, dtype=source.dtype
    )
    target_weights = torch.full(
        (budget,), 1.0 / budget, device=source.device, dtype=source.dtype
    )
    plan = None
    log_plan = None
    mass = None
    for round_index in range(max(1, int(refinement_rounds))):
        cost = metric.compute_cost_matrix(source, supports, query)
        plan, log_plan = unbalanced_sinkhorn(
            cost,
            source_weights,
            target_weights,
            eps=eps,
            rho_s=rho_s,
            rho_t=rho_t,
            iterations=sinkhorn_iterations,
        )
        supports, mass = barycentric_update(plan, source)
        if not torch.isfinite(supports).all():
            # Preserve the deterministic temporal fallback used by the final
            # training path when a finite update cannot be formed.
            chunks = torch.chunk(source, budget, dim=0)
            supports = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
            mass = torch.ones(
                budget, device=source.device, dtype=source.dtype
            ) / float(budget)
            plan = torch.zeros(
                frames, budget, device=source.device, dtype=source.dtype
            )
            start = 0
            for index, chunk in enumerate(chunks):
                plan[start : start + chunk.shape[0], index] = 1.0 / max(1, chunk.shape[0])
                start += chunk.shape[0]
            log_plan = plan.clamp_min(1e-20).log()
            break
        if round_index + 1 < int(refinement_rounds):
            supports = supports.detach()
    assert plan is not None and log_plan is not None and mass is not None
    source_f = source.detach().float()
    supports_f = supports.float()
    distortion = (
        source_f.square().sum(dim=-1, keepdim=True)
        + supports_f.square().sum(dim=-1, keepdim=True).transpose(0, 1)
        - 2.0 * source_f @ supports_f.transpose(0, 1)
    ).clamp_min(0.0) / max(1, source.shape[-1])
    plan_f = log_plan.float().exp()
    loss = (distortion * plan_f).sum() / plan_f.sum().clamp_min(1e-8)
    return supports, mass, loss, plan


def _normalize_signal(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() <= 1:
        return values * 0.0
    centered = values - values.mean()
    scale = centered.std(unbiased=False)
    if float(scale.item()) < 1e-6:
        maximum = centered.abs().max()
        return centered * 0.0 if float(maximum.item()) < 1e-6 else centered / maximum.clamp_min(1e-6)
    return centered / scale.clamp_min(1e-6)


def repair_segment_budgets(
    budgets: Sequence[int],
    sizes: Sequence[int],
    target: int,
) -> list[int]:
    """Repair rounded integer allocations while preserving one support per segment."""

    if not budgets:
        return []
    target = max(1, min(int(target), sum(int(size) for size in sizes)))
    result = [max(0, min(int(budget), int(size))) for budget, size in zip(budgets, sizes)]
    floors = [1 if int(size) > 0 else 0 for size in sizes]
    difference = sum(result) - target
    while difference > 0 and any(budget > floor for budget, floor in zip(result, floors)):
        index = max(
            (i for i, (budget, floor) in enumerate(zip(result, floors)) if budget > floor),
            key=lambda i: result[i] - floors[i],
        )
        result[index] -= 1
        difference -= 1
    difference = target - sum(result)
    while difference > 0 and any(budget < int(size) for budget, size in zip(result, sizes)):
        index = max(
            (i for i, (budget, size) in enumerate(zip(result, sizes)) if budget < int(size)),
            key=lambda i: int(sizes[i]) - result[i],
        )
        result[index] += 1
        difference -= 1
    if difference != 0:
        raise RuntimeError(f"unable to repair segment budgets: {result}, target={target}")
    return result


def allocate_segment_budgets(
    segments: Sequence[torch.Tensor],
    target: int,
    *,
    metric: QueryModulatedMetric,
    query: Optional[torch.Tensor],
    eps: float,
    rho_s: float,
    rho_t: float,
    sinkhorn_iterations: int,
    temperature: float,
    relevance: Optional[torch.Tensor],
    relevance_weight: float,
) -> list[int]:
    """Allocate the stage budget from pilot transport response and relevance."""

    sizes = [int(segment.shape[0]) for segment in segments]
    if not segments:
        return []
    base = [1] * len(segments)
    remaining = int(target) - len(base)
    if remaining <= 0:
        return base
    gains = []
    for segment in segments:
        pilot = _uniform_supports(segment, 1)
        source_weights = torch.full(
            (segment.shape[0],),
            1.0 / segment.shape[0],
            device=segment.device,
            dtype=segment.dtype,
        )
        target_weights = torch.ones(1, device=segment.device, dtype=segment.dtype)
        cost = metric.compute_cost_matrix(segment, pilot, query)
        plan, _ = unbalanced_sinkhorn(
            cost,
            source_weights,
            target_weights,
            eps=eps,
            rho_s=rho_s,
            rho_t=rho_t,
            iterations=sinkhorn_iterations,
        )
        row_mass = plan.sum(dim=1).float()
        profile = row_mass / row_mass.sum().clamp_min(1e-8)
        uniform = profile.new_full(profile.shape, 1.0 / profile.numel())
        gains.append((profile - uniform).abs().mean())
    logits = _normalize_signal(torch.stack(gains))
    if relevance is not None:
        relevance = relevance.flatten().float()
        if relevance.numel() != sum(sizes):
            raise ValueError("segment relevance must match the stage frame count")
        prior = []
        offset = 0
        for size in sizes:
            prior.append(relevance[offset : offset + size].mean())
            offset += size
        logits = logits + float(relevance_weight) * _normalize_signal(torch.stack(prior))
    weights = F.softmax(logits / max(float(temperature), 1e-6), dim=0)
    room = torch.tensor([max(0, size - 1) for size in sizes], device=weights.device)
    allocation = torch.minimum(torch.round(weights * remaining).long(), room)
    budgets = [1 + int(value.item()) for value in allocation]
    order = torch.argsort(weights, descending=True).tolist()
    difference = int(target) - sum(budgets)
    cursor = 0
    while difference > 0:
        index = order[cursor % len(order)]
        if budgets[index] < sizes[index]:
            budgets[index] += 1
            difference -= 1
        cursor += 1
    cursor = 0
    while difference < 0:
        index = order[-(cursor % len(order)) - 1]
        if budgets[index] > 1:
            budgets[index] -= 1
            difference += 1
        cursor += 1
    return repair_segment_budgets(budgets, sizes, target)


class AVIOTTemporalCompressor(nn.Module):
    """Question-conditioned progressive temporal AVIOT compressor."""

    def __init__(
        self,
        embed_dim: int,
        *,
        k_video: int,
        cost_dim: int,
        num_temporal_segments: int,
        min_budget_per_segment: int,
        eps: float,
        rho_s: float,
        rho_t: float,
        sinkhorn_iter: int,
        num_ot_rounds: int,
        lambda_ot: float,
        eta_mass: float,
        progressive_ratios: Sequence[float],
        progressive_round_to: int,
        progressive_max_input_frames: int,
        final_ratio_choices: Sequence[float],
        final_ratio_policy: str,
        allocation_temperature_initial: float,
        allocation_temperature_final: float,
        allocation_temperature_anneal_steps: int,
        segment_prior_alpha: float,
        segment_prior_warmup_steps: int,
        multiscale_blocks: Sequence[int],
        multiscale_parent_prior: Sequence[float],
        multiscale_eps: Sequence[float],
        multiscale_rounds: Sequence[int],
        multiscale_warmup_steps: int,
        regional_transport_weight: float,
        continuity_weight: float,
        continuity_sigma: float,
        gate_hidden_dim: int,
        gate_global_floor: float,
        gate_temperature: float,
        gate_tv_weight: float,
        gate_balance_weight: float,
        gate_balance_target: Sequence[float],
        gate_entropy_weight: float,
        gate_entropy_floor: float,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.k_video = int(k_video)
        self.num_temporal_segments = int(num_temporal_segments)
        self.min_budget_per_segment = int(min_budget_per_segment)
        self.eps = float(eps)
        self.rho_s = float(rho_s)
        self.rho_t = float(rho_t)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.num_ot_rounds = int(num_ot_rounds)
        self.lambda_ot = float(lambda_ot)
        self.eta_mass = float(eta_mass)
        self.progressive_ratios = [float(value) for value in progressive_ratios]
        self.progressive_round_to = int(progressive_round_to)
        self.progressive_max_input_frames = int(progressive_max_input_frames)
        self.final_ratio_choices = [float(value) for value in final_ratio_choices]
        self.final_ratio_policy = str(final_ratio_policy)
        self.allocation_temperature_initial = float(allocation_temperature_initial)
        self.allocation_temperature_final = float(allocation_temperature_final)
        self.allocation_temperature_anneal_steps = int(
            allocation_temperature_anneal_steps
        )
        self.segment_prior_alpha = float(segment_prior_alpha)
        self.segment_prior_warmup_steps = int(segment_prior_warmup_steps)
        if self.final_ratio_policy not in {"random", "fixed_ratio"}:
            raise ValueError("final_ratio_policy must be 'random' or 'fixed_ratio'")
        if not self.final_ratio_choices:
            raise ValueError("final_ratio_choices must not be empty")
        self.metric = QueryModulatedMetric(self.embed_dim, int(cost_dim))
        self.multiscale_stage = AVIOTMultiscaleStage(
            self.embed_dim,
            blocks=tuple(int(value) for value in multiscale_blocks),
            parent_prior_medium=float(multiscale_parent_prior[0]),
            parent_prior_local=float(multiscale_parent_prior[1]),
            eps=tuple(float(value) for value in multiscale_eps),
            ot_rounds=tuple(int(value) for value in multiscale_rounds),
            sinkhorn_iter=self.sinkhorn_iter,
            rho_s=self.rho_s,
            rho_t=self.rho_t,
            warmup_steps=int(multiscale_warmup_steps),
            global_floor=float(gate_global_floor),
            gate_hidden_dim=int(gate_hidden_dim),
            regional_transport_weight=float(regional_transport_weight),
            continuity_weight=float(continuity_weight),
            continuity_sigma=float(continuity_sigma),
            region_pos_num_bands=8,
            gate_type="joint_multiscale_stats",
            gate_temperature=float(gate_temperature),
            require_square=True,
            gate_tv_weight=float(gate_tv_weight),
            gate_balance_weight=float(gate_balance_weight),
            gate_balance_target=tuple(float(value) for value in gate_balance_target),
            gate_balance_mode="anti_collapse",
            gate_global_max=0.60,
            gate_medium_min=0.15,
            gate_local_min=0.15,
            gate_entropy_weight=float(gate_entropy_weight),
            gate_entropy_floor=float(gate_entropy_floor),
        )
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer(
            "allocation_temperature",
            torch.tensor([self.allocation_temperature_initial]),
            persistent=True,
        )
        self.register_buffer("_ratio_sample_counter", torch.tensor(0, dtype=torch.long), persistent=False)
        self._mass_distribution: Optional[torch.Tensor] = None
        self.last_stats: dict[str, object] = {}
        self.last_aux_stats: dict[str, object] = {}

    @staticmethod
    def target_frames_for_ratio(frames: int, ratio: float) -> int:
        if float(ratio) <= 1.0:
            raise ValueError("compression ratio must be greater than one")
        return max(1, min(int(frames), math.ceil(float(frames) / float(ratio))))

    @staticmethod
    def _round_stage_target(value: float, lower: int, upper: int, multiple: int) -> int:
        rounded = int(math.floor(value / max(1, multiple) + 0.5) * max(1, multiple))
        return max(int(lower), min(int(upper), rounded))

    def _stage_targets(self, frames: int, target: int) -> list[int]:
        current = int(frames)
        stages = []
        for fraction in self.progressive_ratios:
            candidate = self._round_stage_target(
                frames * fraction,
                target,
                current,
                self.progressive_round_to,
            )
            if target <= candidate < current:
                stages.append(candidate)
                current = candidate
            if current == target:
                break
        if not stages or stages[-1] != target:
            stages.append(target)
        return stages

    def _sample_ratio(self) -> float:
        if self.final_ratio_policy == "fixed_ratio":
            return float(self.final_ratio_choices[0])
        # Keep the ratio choice independent of the global torch stream while
        # matching the distributed final training path exactly.
        rank = _distributed_rank()
        step = int(self.current_step.detach().cpu().item())
        counter = int(self._ratio_sample_counter.detach().cpu().item())
        self._ratio_sample_counter.add_(1)
        seed = random.getrandbits(63)
        seed ^= (rank + 1) * 0x9E3779B97F4A7C15
        seed ^= (step + 1) * 0xBF58476D1CE4E5B9
        seed ^= (counter + 1) * 0x94D049BB133111EB
        rng = random.Random(seed & ((1 << 63) - 1))
        return self.final_ratio_choices[rng.randrange(len(self.final_ratio_choices))]

    def _scheduled_allocation_temperature(self) -> float:
        if self.allocation_temperature_anneal_steps <= 0:
            return self.allocation_temperature_final
        progress = min(
            1.0,
            max(
                0.0,
                float(self.current_step.item())
                / self.allocation_temperature_anneal_steps,
            ),
        )
        return self.allocation_temperature_initial + (
            self.allocation_temperature_final
            - self.allocation_temperature_initial
        ) * progress

    def update_allocation_temperature(self) -> None:
        self.allocation_temperature.fill_(
            self._scheduled_allocation_temperature()
        )

    def step(self) -> None:
        self.current_step.add_(1)
        self.update_allocation_temperature()

    @property
    def mass_distribution(self) -> Optional[torch.Tensor]:
        return self._mass_distribution

    def aviot_aux_loss(
        self,
        transport_loss: torch.Tensor,
        mass_distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.lambda_ot * transport_loss
        parameter_anchor = loss.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter_anchor = parameter_anchor + parameter.float().sum() * 0.0
        loss = loss + parameter_anchor
        mass = mass_distribution if mass_distribution is not None else self._mass_distribution
        entropy = None
        if mass is not None and mass.numel() > 1:
            safe = torch.nan_to_num(mass.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            probabilities = safe / safe.sum().clamp_min(1e-8)
            negative_entropy = (probabilities * probabilities.clamp_min(1e-8).log()).sum()
            loss = loss + self.eta_mass * negative_entropy
            entropy = float((-negative_entropy).detach().cpu().item())
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_aux_stats = {
            "transport_loss": float(transport_loss.detach().float().item()),
            "lambda_ot": self.lambda_ot,
            "eta_mass": self.eta_mass,
            "mass_count": 0 if mass is None else int(mass.numel()),
            "mass_entropy": entropy,
            "aux_loss": float(loss.detach().float().item()),
        }
        return loss

    def _segments(self, tensor: torch.Tensor, count: Optional[int] = None) -> list[torch.Tensor]:
        frames = int(tensor.shape[0])
        count = min(self.num_temporal_segments, frames) if count is None else min(int(count), frames)
        count = max(1, count)
        base, remainder = divmod(frames, count)
        result = []
        start = 0
        for index in range(count):
            end = start + base + (1 if index < remainder else 0)
            result.append(tensor[start:end])
            start = end
        return result

    def _stage_compress(
        self,
        frame_descriptors: torch.Tensor,
        target: int,
        query: torch.Tensor,
        relevance: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        minimum = max(1, min(self.min_budget_per_segment, target))
        segment_count = min(self.num_temporal_segments, target // minimum, frame_descriptors.shape[0])
        segments = self._segments(frame_descriptors, max(1, segment_count))
        budgets = allocate_segment_budgets(
            segments,
            target,
            metric=self.metric,
            query=query,
            eps=self.eps,
            rho_s=self.rho_s,
            rho_t=self.rho_t,
            sinkhorn_iterations=self.sinkhorn_iter,
            temperature=float(self.allocation_temperature.item()),
            relevance=relevance,
            relevance_weight=self.segment_prior_alpha * min(
                1.0,
                float(self.current_step.item()) / max(1, self.segment_prior_warmup_steps),
            ),
        )
        supports = []
        plans = []
        source_mass = []
        target_mass = []
        losses = []
        frame_offset = 0
        for segment, budget in zip(segments, budgets):
            refined, mass, loss, plan = ot_consolidate_segment(
                segment,
                budget,
                self.metric,
                query=query,
                eps=self.eps,
                rho_s=self.rho_s,
                rho_t=self.rho_t,
                sinkhorn_iterations=self.sinkhorn_iter,
                refinement_rounds=self.num_ot_rounds,
            )
            supports.append(refined)
            target_mass.append(mass)
            full_plan = torch.zeros(
                frame_descriptors.shape[0],
                budget,
                device=frame_descriptors.device,
                dtype=plan.dtype,
            )
            full_plan[frame_offset : frame_offset + segment.shape[0]] = plan
            plans.append(full_plan)
            source_mass.append(
                torch.nan_to_num(
                    plan.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).sum(dim=1).clamp_min(0.0)
            )
            losses.append(loss)
            frame_offset += segment.shape[0]
        return (
            torch.cat(supports, dim=0),
            torch.cat(plans, dim=1),
            budgets,
            torch.cat(source_mass, dim=0) if source_mass else None,
            torch.cat(target_mass, dim=0) if target_mass else None,
            torch.stack(losses).sum() if losses else frame_descriptors.new_zeros((), dtype=torch.float32),
        )

    def compress_video(
        self,
        video_features: torch.Tensor,
        *,
        query: Optional[torch.Tensor] = None,
        query_source: str = "question",
        frame_relevance: Optional[torch.Tensor] = None,
        motion_features: Optional[torch.Tensor] = None,
        extra_stats: Optional[dict[str, object]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if video_features.ndim != 3 or not torch.is_floating_point(video_features):
            raise ValueError("video_features must be a floating-point [T,S,D] tensor")
        frames, spatial_tokens, hidden = video_features.shape
        if hidden != self.embed_dim:
            raise ValueError(f"expected feature dimension {self.embed_dim}, got {hidden}")
        original_dtype = video_features.dtype
        parameter = next(self.parameters(), None)
        compute_dtype = parameter.dtype if parameter is not None else original_dtype
        video_features_for_compute = (
            video_features.to(dtype=compute_dtype)
            if video_features.dtype != compute_dtype
            else video_features
        )
        self.update_allocation_temperature()
        ratio = self._sample_ratio()
        target = self.target_frames_for_ratio(frames, ratio)
        stats = {
            "input_frames": int(frames),
            "target_frames": int(target),
            "output_frames": int(frames),
            "spatial_tokens": int(spatial_tokens),
            "sampled_final_ratio": float(ratio),
            "final_ratio_policy": self.final_ratio_policy,
            "query_source": query_source,
            "query_norm": None if query is None else float(query.detach().float().norm().cpu().item()),
            "progressive_path": [int(frames)],
            "segment_budgets": [],
            "allocation_temperature": float(self.allocation_temperature.item()),
        }
        if extra_stats:
            stats.update(extra_stats)
        zero = video_features.new_zeros((), dtype=torch.float32)
        self._mass_distribution = None
        if frames <= target:
            self.last_stats = stats
            return video_features, zero, stats
        raw_mean = video_features_for_compute.float().mean(dim=1).to(video_features_for_compute.dtype)
        if query is None:
            query = raw_mean.mean(dim=0)
            query_source = "video_mean"
        else:
            query = query.to(device=video_features.device, dtype=compute_dtype)
        relevance = None if frame_relevance is None else frame_relevance.to(video_features.device).flatten().float()
        if relevance is not None and relevance.numel() != frames:
            raise ValueError("frame_relevance must have one value per source frame")
        current_features = video_features_for_compute
        current_descriptors = raw_mean
        current_relevance = relevance
        patch_motion = compute_feature_delta_motion(
            video_features if motion_features is None else motion_features
        )
        stage_losses = []
        stage_descriptions = []
        last_budgets = []
        last_source_mass = None
        for stage_target in self._stage_targets(frames, target):
            if current_features.shape[0] <= stage_target:
                continue
            (
                _,
                global_plan,
                budgets,
                source_mass,
                target_mass,
                global_loss,
            ) = self._stage_compress(
                current_descriptors,
                stage_target,
                query,
                current_relevance,
            )
            segment_sizes = [segment.shape[0] for segment in self._segments(current_descriptors, len(budgets))]
            result = self.multiscale_stage(
                current_features,
                global_plan,
                segment_sizes=segment_sizes,
                segment_budgets=budgets,
                metric=self.metric,
                query=query,
                patch_motion=patch_motion,
                warmup_alpha=min(
                    1.0,
                    float(self.current_step.item()) / max(1, self.multiscale_stage.warmup_steps),
                ),
            )
            current_features = result.output
            current_descriptors = current_features.float().mean(dim=1).to(current_features.dtype)
            if current_relevance is not None:
                conditional = global_plan.float() / global_plan.float().sum(dim=0, keepdim=True).clamp_min(1e-8)
                current_relevance = torch.einsum("tk,t->k", conditional, current_relevance)
            stage_losses.append(global_loss + result.auxiliary_loss)
            last_budgets = [int(value) for value in budgets]
            last_source_mass = source_mass
            stats["progressive_path"].append(int(current_features.shape[0]))
            stage_descriptions.append(
                {
                    "target_frames": int(stage_target),
                    "segment_budgets": last_budgets,
                    "transport_loss": float(result.auxiliary_loss.detach().float().cpu().item()),
                }
            )
            # The target-column masses are the same quantities used by the
            # auxiliary mass regularizer in the original final path.
            self._mass_distribution = (
                None if target_mass is None else target_mass.detach()
            )
        if not stage_losses:
            self.last_stats = stats
            return video_features, zero, stats
        transport_loss = torch.stack([item.float() for item in stage_losses]).mean()
        parameter_anchor = transport_loss.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter_anchor = parameter_anchor + parameter.float().sum() * 0.0
        transport_loss = transport_loss + parameter_anchor
        compressed = current_features + parameter_anchor.to(current_features.dtype)
        stats.update(
            {
                "output_frames": int(compressed.shape[0]),
                "compressed": True,
                "transport_loss": float(transport_loss.detach().float().cpu().item()),
                "segment_budgets": last_budgets,
                "progressive_stages": stage_descriptions,
                "effective_segments": len(last_budgets),
                "mass_count": 0 if self._mass_distribution is None else int(self._mass_distribution.numel()),
                "source_temporal_group_mass": (
                    None if last_source_mass is None else [float(value) for value in last_source_mass.detach().cpu().tolist()]
                ),
            }
        )
        self.last_stats = stats
        return compressed.to(original_dtype), transport_loss, stats
