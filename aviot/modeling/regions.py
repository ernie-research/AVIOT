from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionLayout:
    height: int
    width: int
    block: int
    patch_ids: torch.Tensor
    row_ids: torch.Tensor
    col_ids: torch.Tensor

    @property
    def num_regions(self) -> int:
        return int(self.patch_ids.shape[0])

    @property
    def patches_per_region(self) -> int:
        return int(self.patch_ids.shape[1])

    def parent_ids(self, parent: "RegionLayout") -> torch.Tensor:
        if self.height != parent.height or self.width != parent.width:
            raise ValueError("child and parent region layouts must use the same grid")
        if parent.block < self.block or parent.block % self.block != 0:
            raise ValueError("parent block must be an integer multiple of child block")
        parent_cols = self.width // parent.block
        center_rows = self.row_ids * self.block + self.block // 2
        center_cols = self.col_ids * self.block + self.block // 2
        return (center_rows // parent.block) * parent_cols + center_cols // parent.block


def build_region_layout(height: int, width: int, block: int) -> RegionLayout:
    height = int(height)
    width = int(width)
    block = int(block)
    if block <= 0:
        raise ValueError(f"region block must be positive, got {block}")
    if height <= 0 or width <= 0 or height % block or width % block:
        raise ValueError(
            f"grid {(height, width)} must be positive and divisible by block={block}"
        )

    grid = torch.arange(height * width, dtype=torch.long).view(height, width)
    region_rows = height // block
    region_cols = width // block
    patch_ids = (
        grid.view(region_rows, block, region_cols, block)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(region_rows * region_cols, block * block)
    )
    row_ids = torch.arange(region_rows, dtype=torch.long).repeat_interleave(region_cols)
    col_ids = torch.arange(region_cols, dtype=torch.long).repeat(region_rows)
    return RegionLayout(height, width, block, patch_ids, row_ids, col_ids)


def region_descriptors(
    features: torch.Tensor,
    layout: RegionLayout,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(f"features must be [T,S,D], got {tuple(features.shape)}")
    if int(features.shape[1]) != layout.height * layout.width:
        raise ValueError(
            f"feature patch count {features.shape[1]} does not match layout "
            f"{layout.height}x{layout.width}"
        )
    patch_ids = layout.patch_ids.to(device=features.device)
    gathered = features.float()[:, patch_ids, :]
    descriptors = gathered.mean(dim=2).permute(1, 0, 2).contiguous()
    if normalize:
        descriptors = F.layer_norm(descriptors, (descriptors.shape[-1],))
    return descriptors


def compute_feature_delta_motion(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(f"features must be [T,S,D], got {tuple(features.shape)}")
    with torch.no_grad():
        values = features.detach().float()
        if int(values.shape[0]) <= 1:
            return torch.zeros(values.shape[1], device=values.device, dtype=torch.float32)
        normalized = F.normalize(values, dim=-1, eps=1e-6)
        cosine = (normalized[1:] * normalized[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cosine).mean(dim=0)


def apply_regional_plans(
    features: torch.Tensor,
    plans: torch.Tensor,
    layout: RegionLayout,
) -> tuple[torch.Tensor, int]:
    if features.dim() != 3:
        raise ValueError(f"features must be [T,S,D], got {tuple(features.shape)}")
    if plans.dim() != 3:
        raise ValueError(f"plans must be [R,T,K], got {tuple(plans.shape)}")
    frames, patches, _ = features.shape
    if patches != layout.height * layout.width:
        raise ValueError("feature patch count does not match region layout")
    if plans.shape[0] != layout.num_regions or plans.shape[1] != frames:
        raise ValueError("regional plan shape does not match layout or feature frames")

    patch_ids = layout.patch_ids.to(device=features.device)
    regional_features = features.float()[:, patch_ids, :].permute(1, 0, 2, 3)
    plan = plans.float()
    mass = plan.sum(dim=1)
    positive = mass > 0
    numerator = torch.einsum("rtk,rtad->rkad", plan, regional_features)
    output = numerator / torch.where(positive, mass, torch.ones_like(mass))[..., None, None]
    fallback = regional_features.mean(dim=1)[:, None, :, :]
    output = torch.where(positive[..., None, None], output, fallback)

    ordered = output.permute(1, 0, 2, 3).reshape(plans.shape[2], patches, features.shape[2])
    inverse = torch.argsort(patch_ids.flatten())
    ordered = ordered.index_select(1, inverse)
    if not torch.isfinite(ordered).all():
        raise FloatingPointError("regional plan application produced non-finite output")
    return ordered.to(features.dtype), int((~positive).sum().item())
