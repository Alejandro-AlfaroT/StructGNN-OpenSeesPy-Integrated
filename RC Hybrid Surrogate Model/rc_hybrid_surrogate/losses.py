"""Masked sequence objectives."""

import torch
import torch.nn.functional as functional


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Huber loss averaged over valid timesteps and response components."""
    per_component = functional.smooth_l1_loss(
        prediction, target, reduction="none", beta=beta
    )
    valid = mask.unsqueeze(-1).expand_as(per_component)
    return per_component[valid].mean()
