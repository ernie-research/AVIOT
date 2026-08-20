import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import (
    RegionLayout,
    apply_regional_plans,
    build_region_layout,
    region_descriptors,
)


@dataclass
class MultiscaleStageResult:
    output: torch.Tensor
    global_output: torch.Tensor
    medium_output: torch.Tensor
    local_output: torch.Tensor
    medium_plans: torch.Tensor
    local_plans: torch.Tensor
    local_parent_ids: torch.Tensor
    gate: torch.Tensor
    auxiliary_loss: torch.Tensor
    stats: Dict[str, object]


def _parse_tuple(values: Sequence, length: int, cast, name: str):
    parsed = tuple(cast(value) for value in values)
    if len(parsed) != length:
        raise ValueError(f"{name} must contain {length} values, got {parsed}")
    return parsed


def _column_conditionals_from_plan(plan: torch.Tensor) -> torch.Tensor:
    plan_f = plan.float()
    if not torch.isfinite(plan_f).all() or bool((plan_f < 0).any()):
        raise ValueError("global/parent plan must contain finite non-negative values")
    mass = plan_f.sum(dim=-2, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("global/parent plan must have positive mass in every slot")
    return plan_f / mass


def _regional_cost(metric, source: torch.Tensor, support: torch.Tensor, query) -> torch.Tensor:
    source_phi = metric.phi(source)
    support_phi = metric.phi(support)
    if query is not None:
        query_value = query
        if query_value.dim() == 1:
            query_value = query_value.unsqueeze(0)
        weight = F.softplus(metric.h_theta(query_value.to(source_phi.dtype))).reshape(1, 1, -1)
        source_phi = source_phi * weight
        support_phi = support_phi * weight
    source_sq = source_phi.square().sum(dim=-1, keepdim=True)
    support_sq = support_phi.square().sum(dim=-1).unsqueeze(-2)
    return (source_sq + support_sq - 2 * torch.einsum("rtd,rkd->rtk", source_phi, support_phi)).clamp_min(0)


def _batched_unbalanced_sinkhorn(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    eps: float,
    rho_s: float,
    rho_t: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized log-domain unbalanced Sinkhorn updates."""
    safe_eps = max(float(eps), 1e-6)
    safe_range = safe_eps * 100.0
    with torch.no_grad():
        maximum = cost.detach().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(
        maximum > safe_range,
        safe_range / (maximum + 1e-8),
        torch.ones_like(maximum),
    )
    cost = cost * scale

    log_a = torch.log(source_weights.clamp(min=1e-20))
    log_m = torch.log(target_weights.clamp(min=1e-20))
    f = torch.zeros_like(source_weights)
    g = torch.zeros_like(target_weights)
    kappa_s = float(rho_s) / (float(rho_s) + safe_eps)
    kappa_t = float(rho_t) / (float(rho_t) + safe_eps)
    for _ in range(max(1, int(max_iter))):
        row_lse = torch.logsumexp((g.unsqueeze(-2) - cost) / safe_eps, dim=-1)
        f = kappa_s * (-safe_eps * row_lse + safe_eps * log_a) + (1.0 - kappa_s) * f
        col_lse = torch.logsumexp((f.unsqueeze(-1) - cost) / safe_eps, dim=-2)
        g = kappa_t * (-safe_eps * col_lse + safe_eps * log_m) + (1.0 - kappa_t) * g
    log_plan = (f.unsqueeze(-1) + g.unsqueeze(-2) - cost) / safe_eps
    return torch.exp(log_plan), log_plan


def _broadcast_region_values(values: torch.Tensor, layout: RegionLayout) -> torch.Tensor:
    # values [R,K] -> [K,S]
    patch_ids = layout.patch_ids.to(values.device)
    expanded = values[:, :, None].expand(-1, -1, layout.patches_per_region)
    flattened = expanded.permute(1, 0, 2).reshape(values.shape[1], -1)
    inverse = torch.argsort(patch_ids.flatten())
    return flattened.index_select(1, inverse)


def _plan_entropy(plans: torch.Tensor) -> torch.Tensor:
    conditional = _column_conditionals_from_plan(plans)
    entropy = -(conditional * torch.log(conditional.clamp_min(1e-20))).sum(dim=-2)
    denom = math.log(max(2, plans.shape[-2]))
    return entropy / denom


def _adjacent_pairs(layout: RegionLayout, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = layout.height // layout.block
    cols = layout.width // layout.block
    grid = torch.arange(rows * cols, device=device).view(rows, cols)
    left = torch.cat([grid[:, :-1].reshape(-1), grid[:-1, :].reshape(-1)])
    right = torch.cat([grid[:, 1:].reshape(-1), grid[1:, :].reshape(-1)])
    return left, right


def _js_continuity(
    plans: torch.Tensor,
    layout: RegionLayout,
    boundary_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if layout.num_regions <= 1:
        return plans.new_zeros((), dtype=torch.float32)
    conditional = _column_conditionals_from_plan(plans)
    first, second = _adjacent_pairs(layout, plans.device)
    p = conditional.index_select(0, first).clamp_min(1e-20)
    q = conditional.index_select(0, second).clamp_min(1e-20)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=1)
        + (q * (q.log() - midpoint.log())).sum(dim=1)
    )
    if boundary_weights is None:
        return divergence.mean()
    weights = boundary_weights.detach().float().view(-1, 1)
    return (divergence * weights).sum() / weights.sum().clamp_min(1e-8) / divergence.shape[1]


class AVIOTMultiscaleStage(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        blocks: Sequence[int] = (27, 9, 3),
        parent_prior_medium: float = 0.75,
        parent_prior_local: float = 0.5,
        eps: Sequence[float] = (0.1, 0.12, 0.15),
        ot_rounds: Sequence[int] = (3, 2, 1),
        sinkhorn_iter: int = 20,
        rho_s: float = 0.5,
        rho_t: float = 5.0,
        warmup_steps: int = 500,
        global_floor: float = 0.2,
        gate_hidden_dim: int = 16,
        regional_transport_weight: float = 1.0,
        continuity_weight: float = 0.01,
        continuity_sigma: float = 0.20,
        region_pos_enable: bool = True,
        region_pos_num_bands: int = 8,
        gate_type: str = "joint_multiscale_stats",
        gate_temperature: float = 1.0,
        require_square: bool = True,
        gate_tv_weight: float = 0.001,
        gate_balance_weight: float = 0.001,
        gate_balance_target: Sequence[float] = (0.35, 0.45, 0.20),
        gate_balance_mode: str = "target_mse",
        gate_global_max: float = 0.60,
        gate_medium_min: float = 0.15,
        gate_local_min: float = 0.15,
        gate_entropy_weight: float = 0.0,
        gate_entropy_floor: float = 0.85,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.blocks = _parse_tuple(blocks, 3, int, "blocks")
        if self.blocks[0] <= 0 or any(value <= 0 for value in self.blocks):
            raise ValueError("multiscale blocks must be positive")
        if self.blocks[0] % self.blocks[1] or self.blocks[1] % self.blocks[2]:
            raise ValueError("multiscale blocks must form a divisible hierarchy")
        # The released configuration evaluates every local region.
        self.parent_prior_medium = float(parent_prior_medium)
        self.parent_prior_local = float(parent_prior_local)
        if not 0.0 <= self.parent_prior_medium <= 1.0:
            raise ValueError("parent_prior_medium must be in [0,1]")
        if not 0.0 <= self.parent_prior_local <= 1.0:
            raise ValueError("parent_prior_local must be in [0,1]")
        self.eps = _parse_tuple(eps, 3, float, "eps")
        self.ot_rounds = _parse_tuple(ot_rounds, 3, int, "ot_rounds")
        if any(value <= 0 for value in self.ot_rounds):
            raise ValueError(f"ot_rounds values must be > 0, got {self.ot_rounds!r}")
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.rho_s = float(rho_s)
        self.rho_t = float(rho_t)
        self.warmup_steps = max(0, int(warmup_steps))
        if not 0 <= float(global_floor) < 1:
            raise ValueError("global_floor must be in [0,1)")
        self.global_floor = float(global_floor)
        self.regional_transport_weight = float(regional_transport_weight)
        self.continuity_weight = float(continuity_weight)
        self.continuity_sigma = float(continuity_sigma)
        if not math.isfinite(self.continuity_sigma) or self.continuity_sigma <= 0:
            raise ValueError("continuity_sigma must be finite and > 0")
        self.region_pos_enable = bool(region_pos_enable)
        self.region_pos_num_bands = int(region_pos_num_bands)
        if self.region_pos_num_bands <= 0:
            raise ValueError("region_pos_num_bands must be > 0")
        self.gate_type = str(gate_type).strip().lower()
        if self.gate_type != "joint_multiscale_stats":
            raise ValueError("the released model uses joint_multiscale_stats gating")
        self.gate_temperature = float(gate_temperature)
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be finite and > 0")
        self.require_square = bool(require_square)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_tv_weight = float(gate_tv_weight)
        self.gate_balance_weight = float(gate_balance_weight)
        self.gate_balance_mode = str(gate_balance_mode).strip().lower()
        if self.gate_balance_mode not in {"target_mse", "anti_collapse"}:
            raise ValueError("gate_balance_mode must be 'target_mse' or 'anti_collapse'")
        self.gate_global_max = float(gate_global_max)
        self.gate_medium_min = float(gate_medium_min)
        self.gate_local_min = float(gate_local_min)
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.gate_entropy_floor = float(gate_entropy_floor)
        for name, value in (
            ("regional_transport_weight", self.regional_transport_weight),
            ("continuity_weight", self.continuity_weight),
            ("gate_tv_weight", self.gate_tv_weight),
            ("gate_balance_weight", self.gate_balance_weight),
            ("gate_entropy_weight", self.gate_entropy_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if not 0.0 <= self.gate_entropy_floor <= 1.0:
            raise ValueError("gate_entropy_floor must be finite and in [0,1]")
        gate_bounds = (
            self.gate_global_max,
            self.gate_medium_min,
            self.gate_local_min,
        )
        if (
            any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in gate_bounds)
            or self.gate_global_max < self.global_floor
            or self.global_floor + self.gate_medium_min + self.gate_local_min > 1.0
        ):
            raise ValueError("gate bounds are infeasible with global_floor")
        target_values = tuple(float(value) for value in gate_balance_target)
        if (
            len(target_values) != 3
            or any(value < 0 for value in target_values)
            or not math.isclose(sum(target_values), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("gate_balance_target must be three non-negative values summing to one")
        target = torch.tensor(target_values, dtype=torch.float32)
        self.register_buffer("gate_balance_target", target, persistent=True)

        if self.region_pos_enable:
            self.region_position = nn.Linear(4 * self.region_pos_num_bands, self.embed_dim, bias=False)
            nn.init.zeros_(self.region_position.weight)
        else:
            self.region_position = None
        self.joint_gate_mlp = nn.Sequential(
            nn.LayerNorm(21),
            nn.Linear(21, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 3),
        )
        nn.init.normal_(self.joint_gate_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.joint_gate_mlp[-1].bias)

    @staticmethod
    def _route_entropy(gate: torch.Tensor) -> torch.Tensor:
        route_count = gate.shape[0]
        if route_count <= 1:
            return gate.new_ones((), dtype=torch.float32)
        probabilities = gate.float().clamp_min(1e-20)
        probabilities = probabilities / probabilities.sum(dim=0, keepdim=True).clamp_min(1e-20)
        return -(probabilities * probabilities.log()).sum(dim=0).mean() / math.log(route_count)

    def _gate_regularization(
        self,
        gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        usage = gate.mean(dim=(1, 2))
        entropy = self._route_entropy(gate)
        violations = torch.stack([
            F.relu(usage[0] - self.gate_global_max),
            F.relu(self.gate_medium_min - usage[1]),
            F.relu(self.gate_local_min - usage[2]),
        ])
        if self.gate_balance_mode == "target_mse":
            balance = (usage - self.gate_balance_target).square().sum()
            entropy_barrier = gate.new_zeros((), dtype=torch.float32)
        else:
            balance = violations.square().mean()
            entropy_barrier = F.relu(self.gate_entropy_floor - entropy).square()
        return {
            "balance": balance,
            "entropy_barrier": entropy_barrier,
            "usage": usage,
            "entropy": entropy,
        }

    @staticmethod
    def _relative_rms(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        difference = (first.float() - second.float()).square().mean(dim=-1).clamp_min(1e-12).sqrt()
        scale = 0.5 * (
            first.float().square().mean(dim=-1).clamp_min(1e-12).sqrt()
            + second.float().square().mean(dim=-1).clamp_min(1e-12).sqrt()
        )
        return (difference / scale.clamp_min(1e-6)).clamp(0.0, 5.0)

    def _joint_gate_features(
        self,
        branches: torch.Tensor,
        entropy: torch.Tensor,
        query: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        if branches.dim() != 4 or branches.shape[0] != 3:
            raise ValueError("joint gate branches must have shape [3,K,S,D]")
        if entropy.shape != branches.shape[:3]:
            raise ValueError("joint gate entropy must match branch scale/slot/patch dimensions")

        branch_f = branches.detach().float()
        entropy_f = entropy.detach().float()
        scale_motion = branch_f.new_zeros(branch_f.shape[:3])
        if branch_f.shape[1] > 1:
            adjacent = self._relative_rms(branch_f[:, 1:], branch_f[:, :-1])
            scale_motion[:, 1:] = adjacent
            scale_motion[:, 0] = adjacent[:, 0]

        branch_mean = branch_f.mean(dim=0, keepdim=True)
        relative_l2 = self._relative_rms(branch_f, branch_mean.expand_as(branch_f))
        cosine_distance = (
            1.0
            - F.cosine_similarity(branch_f, branch_mean.expand_as(branch_f), dim=-1, eps=1e-6)
        ).clamp(0.0, 2.0)

        if query is None or query.shape[-1] != branch_f.shape[-1]:
            query_similarity = branch_f.new_zeros(branch_f.shape[:3])
        else:
            query_value = query.detach().float().reshape(-1, query.shape[-1]).mean(dim=0)
            query_similarity = F.cosine_similarity(
                branch_f,
                query_value.view(1, 1, 1, -1),
                dim=-1,
                eps=1e-6,
            ).clamp(-1.0, 1.0)

        branch_rms = branch_f.square().mean(dim=-1).clamp_min(1e-12).sqrt()
        shared_rms = branch_rms.mean(dim=0, keepdim=True).clamp_min(1e-6)
        relative_log_rms = torch.log(branch_rms / shared_rms).clamp(-5.0, 5.0)
        scale_identity = torch.linspace(
            -1.0, 1.0, 3, device=branches.device, dtype=torch.float32
        ).view(3, 1, 1).expand_as(scale_motion)

        per_scale = torch.stack(
            [
                scale_motion,
                entropy_f.clamp(0.0, 1.0),
                relative_l2,
                cosine_distance,
                query_similarity,
                relative_log_rms,
                scale_identity,
            ],
            dim=-1,
        )
        features = per_scale.permute(1, 2, 0, 3).flatten(-2).clamp(-5.0, 5.0)
        diagnostics = {
            "scale_motion_mean": [
                float(value) for value in scale_motion.detach().mean(dim=(1, 2)).cpu().tolist()
            ],
            "branch_relative_l2_mean": [
                float(value) for value in relative_l2.detach().mean(dim=(1, 2)).cpu().tolist()
            ],
            "branch_cosine_distance_mean": [
                float(value) for value in cosine_distance.detach().mean(dim=(1, 2)).cpu().tolist()
            ],
            "branch_query_similarity_mean": [
                float(value) for value in query_similarity.detach().mean(dim=(1, 2)).cpu().tolist()
            ],
        }
        return features, diagnostics

    def _position_features(self, layout: RegionLayout, device: torch.device) -> torch.Tensor:
        rows = layout.height // layout.block
        cols = layout.width // layout.block
        y = (layout.row_ids.to(device).float() + 0.5) / rows
        x = (layout.col_ids.to(device).float() + 0.5) / cols
        frequencies = torch.arange(1, self.region_pos_num_bands + 1, device=device, dtype=torch.float32)
        y_angles = 2 * math.pi * y[:, None] * frequencies[None, :]
        x_angles = 2 * math.pi * x[:, None] * frequencies[None, :]
        return torch.cat([y_angles.sin(), y_angles.cos(), x_angles.sin(), x_angles.cos()], dim=-1)

    def _layout_for_level(self, level: int) -> RegionLayout:
        if int(level) not in (0, 1, 2):
            raise ValueError(f"AVIOT multiscale level must be 0, 1, or 2, got {level}")
        return build_region_layout(self.blocks[0], self.blocks[0], self.blocks[int(level)])

    def _feature_boundary_weights(self, features: torch.Tensor, layout: RegionLayout) -> torch.Tensor:
        patch_ids = layout.patch_ids.to(features.device)
        descriptors = features.detach().float()[:, patch_ids, :].mean(dim=(0, 2))
        descriptors = F.normalize(descriptors, dim=-1, eps=1e-6)
        first, second = _adjacent_pairs(layout, features.device)
        similarity = (descriptors.index_select(0, first) * descriptors.index_select(0, second)).sum(dim=-1)
        distance = (1.0 - similarity).clamp_min(0.0)
        return torch.exp(-distance / self.continuity_sigma).detach()

    def _motion_boundary_weights(self, patch_motion: torch.Tensor, side: int) -> torch.Tensor:
        motion = patch_motion.detach().float().view(int(side), int(side))
        differences = torch.cat([
            (motion[:, 1:] - motion[:, :-1]).abs().reshape(-1),
            (motion[1:, :] - motion[:-1, :]).abs().reshape(-1),
        ])
        return torch.exp(-differences / self.continuity_sigma).detach()

    def _descriptors(self, features: torch.Tensor, layout: RegionLayout) -> torch.Tensor:
        descriptors = region_descriptors(features, layout, normalize=True)
        if self.region_position is None:
            return descriptors
        position = self.region_position(
            self._position_features(layout, features.device).to(self.region_position.weight.dtype)
        )
        return descriptors.to(position.dtype) + position[:, None, :]

    def _solve_level(
        self,
        descriptors: torch.Tensor,
        parent_plans: torch.Tensor,
        segment_sizes: Sequence[int],
        segment_budgets: Sequence[int],
        metric,
        query: Optional[torch.Tensor],
        *,
        beta: float,
        eps: float,
        rounds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[Dict[str, object]]]:
        regions, frames, _ = descriptors.shape
        total_slots = sum(segment_budgets)
        full_plans = descriptors.new_zeros((regions, frames, total_slots), dtype=torch.float32)
        losses = []
        diagnostics = []
        frame_offset = 0
        slot_offset = 0
        for size, budget in zip(segment_sizes, segment_budgets):
            source = descriptors[:, frame_offset:frame_offset + size, :]
            parent = parent_plans[:, frame_offset:frame_offset + size, slot_offset:slot_offset + budget]
            if size == budget:
                identity = torch.eye(size, device=descriptors.device, dtype=torch.float32).expand(regions, -1, -1)
                full_plans[:, frame_offset:frame_offset + size, slot_offset:slot_offset + budget] = identity
                losses.append(descriptors.new_zeros((), dtype=torch.float32))
                diagnostics.append({"iterations": 0, "residual": 0.0, "converged": True})
                frame_offset += size
                slot_offset += budget
                continue
            parent_conditional = _column_conditionals_from_plan(parent.detach())
            parent_support = torch.einsum("rtk,rtd->rkd", parent_conditional, source.float()).to(source.dtype)
            indices = torch.linspace(
                0, size - 1, budget, device=source.device, dtype=torch.float32
            ).round().to(torch.long)
            baseline_support = source.index_select(1, indices)
            support = (
                float(beta) * parent_support.float()
                + (1.0 - float(beta)) * baseline_support.float()
            ).to(source.dtype)
            source_weights = torch.full((regions, size), 1.0 / size, device=source.device, dtype=torch.float32)
            target_weights = torch.full((regions, budget), 1.0 / budget, device=source.device, dtype=torch.float32)
            plan = None
            log_plan = None
            for round_index in range(max(1, int(rounds))):
                cost = _regional_cost(metric, source, support, query)
                plan, log_plan = _batched_unbalanced_sinkhorn(
                    cost,
                    source_weights,
                    target_weights,
                    eps=eps,
                    rho_s=self.rho_s,
                    rho_t=self.rho_t,
                    max_iter=self.sinkhorn_iter,
                )
                conditional = plan / plan.sum(dim=-2, keepdim=True).clamp_min(1e-8)
                support = torch.einsum("rtk,rtd->rkd", conditional, source.float()).to(source.dtype)
                if round_index < int(rounds) - 1:
                    support = support.detach()
            assert plan is not None and log_plan is not None
            full_plans[:, frame_offset:frame_offset + size, slot_offset:slot_offset + budget] = plan.float()
            source_f = source.detach().float()
            support_f = support.float()
            distortion = (
                source_f.square().sum(dim=-1, keepdim=True)
                + support_f.square().sum(dim=-1).unsqueeze(-2)
                - 2 * torch.einsum("rtd,rkd->rtk", source_f, support_f)
            ).clamp_min(0.0) / max(1, source.shape[-1])
            plan_f = torch.exp(log_plan.float())
            losses.append(
                ((distortion * plan_f).sum(dim=(-2, -1)) / plan_f.sum(dim=(-2, -1)).clamp_min(1e-8)).mean()
            )
            diagnostics.append({"iterations": self.sinkhorn_iter})
            frame_offset += size
            slot_offset += budget
        return full_plans, torch.stack(losses).mean(), diagnostics

    def forward(
        self,
        features: torch.Tensor,
        global_plan: torch.Tensor,
        *,
        segment_sizes: Sequence[int],
        segment_budgets: Sequence[int],
        metric,
        query: Optional[torch.Tensor],
        patch_motion: torch.Tensor,
        warmup_alpha: float,
    ) -> MultiscaleStageResult:
        if features.dim() != 3 or features.shape[-1] != self.embed_dim:
            raise ValueError(f"features must be [T,S,{self.embed_dim}], got {tuple(features.shape)}")
        frames, patches, _ = features.shape
        side = math.isqrt(int(patches))
        if side * side != int(patches) or side != self.blocks[0]:
            raise ValueError(
                f"AVIOT multiscale transport requires a {self.blocks[0]}x{self.blocks[0]} "
                f"grid, got S={patches}"
            )
        sizes = [int(value) for value in segment_sizes]
        budgets = [int(value) for value in segment_budgets]
        if len(sizes) != len(budgets) or sum(sizes) != frames or sum(budgets) != global_plan.shape[1]:
            raise ValueError("global segment metadata is inconsistent with features/plan")
        if any(size <= 0 or budget <= 0 or budget > size for size, budget in zip(sizes, budgets)):
            raise ValueError("each global segment budget must satisfy 0 < budget <= size")
        if global_plan.shape != (frames, sum(budgets)):
            raise ValueError("global plan shape is inconsistent with segment metadata")
        if patch_motion.shape != (patches,) or not torch.isfinite(patch_motion).all():
            raise ValueError("patch_motion must be a finite vector matching the patch grid")

        global_conditional = _column_conditionals_from_plan(global_plan)
        global_output = torch.einsum("tk,tsd->ksd", global_conditional, features.float()).to(features.dtype)
        medium_layout = build_region_layout(side, side, self.blocks[1])
        local_layout = build_region_layout(side, side, self.blocks[2])
        medium_descriptors = self._descriptors(features, medium_layout)
        global_parents = global_conditional.unsqueeze(0).expand(medium_layout.num_regions, -1, -1)
        medium_plans, medium_transport, medium_diagnostics = self._solve_level(
            medium_descriptors, global_parents, sizes, budgets, metric,
            query,
            beta=self.parent_prior_medium,
            eps=self.eps[1], rounds=self.ot_rounds[1],
        )
        medium_output, medium_zero = apply_regional_plans(features, medium_plans, medium_layout)

        local_parent_ids = local_layout.parent_ids(medium_layout).to(features.device)
        active_ids = torch.arange(local_layout.num_regions, device=features.device, dtype=torch.long)
        all_local_descriptors = self._descriptors(features, local_layout)
        local_parents = medium_plans.index_select(0, local_parent_ids)
        local_plans, local_transport, local_diagnostics = self._solve_level(
            all_local_descriptors, local_parents, sizes, budgets, metric,
            query,
            beta=self.parent_prior_local,
            eps=self.eps[2], rounds=self.ot_rounds[2],
        )
        local_output, local_zero = apply_regional_plans(features, local_plans, local_layout)

        global_entropy = _plan_entropy(global_conditional.unsqueeze(0)).expand(patches, -1).t()
        medium_entropy = _broadcast_region_values(_plan_entropy(medium_plans), medium_layout)
        local_entropy = _broadcast_region_values(_plan_entropy(local_plans), local_layout)
        entropy = torch.stack([global_entropy, medium_entropy, local_entropy], dim=0)
        branches = torch.stack([global_output.float(), medium_output.float(), local_output.float()], dim=0)
        gate_diagnostics: Dict[str, object] = {}
        gate_inputs, gate_diagnostics = self._joint_gate_features(branches, entropy, query)
        logits = self.joint_gate_mlp(
            gate_inputs.to(self.joint_gate_mlp[1].weight.dtype)
        ).permute(2, 0, 1).float()
        logits = logits / self.gate_temperature
        probabilities = F.softmax(logits, dim=0)
        gate = probabilities * (1.0 - self.global_floor)
        gate[0] = gate[0] + self.global_floor
        full_output = (gate[..., None] * branches).sum(dim=0)
        alpha_value = max(0.0, min(1.0, float(warmup_alpha)))
        output = global_output.float() + alpha_value * (full_output - global_output.float())

        continuity = 0.5 * (
            _js_continuity(medium_plans, medium_layout, self._feature_boundary_weights(features, medium_layout))
            + _js_continuity(local_plans, local_layout, self._feature_boundary_weights(features, local_layout))
        )
        gate_grid = gate.view(3, gate.shape[1], side, side)
        gate_differences = torch.cat([
            (gate_grid[:, :, :, 1:] - gate_grid[:, :, :, :-1]).abs().reshape(3, gate.shape[1], -1),
            (gate_grid[:, :, 1:, :] - gate_grid[:, :, :-1, :]).abs().reshape(3, gate.shape[1], -1),
        ], dim=-1)
        motion_weights = self._motion_boundary_weights(patch_motion, side).view(1, 1, -1)
        gate_tv = (gate_differences * motion_weights).sum() / (
            motion_weights.sum().clamp_min(1e-8) * gate_differences.shape[0] * gate_differences.shape[1]
        )
        gate_regularization = self._gate_regularization(gate)
        balance = gate_regularization["balance"]
        entropy_barrier = gate_regularization["entropy_barrier"]
        balance_usage = gate_regularization["usage"]
        active_probabilities = probabilities
        gate_softmax_entropy = -(
            active_probabilities * torch.log(active_probabilities.clamp_min(1e-20))
        ).sum(dim=0).mean() / math.log(3.0)
        active_logits = logits
        gate_logit_spread = (
            active_logits.max(dim=0).values - active_logits.min(dim=0).values
        ).mean()
        regional_transport = 0.5 * (medium_transport + local_transport)
        auxiliary = (
            self.regional_transport_weight * regional_transport
            + self.continuity_weight * continuity
            + self.gate_tv_weight * gate_tv
            + self.gate_balance_weight * balance
            + self.gate_entropy_weight * entropy_barrier
        )
        for name, value in (
            ("output", output), ("auxiliary_loss", auxiliary), ("gate", gate),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"AVIOT multiscale output {name} is non-finite")
        stats = {
            "solver": "unbalanced_sinkhorn",
            "warmup_alpha": alpha_value,
            "segment_sizes": sizes,
            "segment_budgets": budgets,
            "medium_plan_count": int(medium_layout.num_regions),
            "local_plan_count": int(local_layout.num_regions),
            "local_region_count": int(active_ids.numel()),
            "regional_transport_loss": float(regional_transport.detach().item()),
            "continuity_loss": float(continuity.detach().item()),
            "gate_tv_loss": float(gate_tv.detach().item()),
            "gate_balance_loss": float(balance.detach().item()),
            "gate_entropy_barrier_loss": float(entropy_barrier.detach().item()),
            "gate_balance_mode": self.gate_balance_mode,
            "gate_type": self.gate_type,
            "gate_softmax_entropy": float(gate_softmax_entropy.detach().item()),
            "gate_logit_spread": float(gate_logit_spread.detach().item()),
            "gate_balance_usage": [float(value) for value in balance_usage.detach().cpu().tolist()],
            "gate_entropy": float(gate_regularization["entropy"].detach().item()),
            "gate_usage": [float(value) for value in gate.detach().mean(dim=(1, 2)).cpu().tolist()],
            "zero_column_count": int(medium_zero + local_zero),
            "medium_solver": medium_diagnostics,
            "local_solver": local_diagnostics,
        }
        stats.update(gate_diagnostics)
        return MultiscaleStageResult(
            output=output.to(features.dtype),
            global_output=global_output,
            medium_output=medium_output,
            local_output=local_output,
            medium_plans=medium_plans,
            local_plans=local_plans,
            local_parent_ids=local_parent_ids,
            gate=gate,
            auxiliary_loss=auxiliary,
            stats=stats,
        )
