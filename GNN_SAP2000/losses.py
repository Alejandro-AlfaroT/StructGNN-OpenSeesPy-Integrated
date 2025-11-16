# Self-defined loss class
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
import torch
import torch.nn as nn


def _masked_abs(pred, target, threshold, dim_weights=None, reduction="sum"):
    """
    L1 with mask (|target| > threshold). Optional per-dimension weights.
    pred, target: [*, D]
    dim_weights:  [D] or None
    reduction: "sum" or "mean"
    """
    if threshold is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        mask = target.abs() > threshold

    diff = (pred - target).abs()  # [*, D]

    if dim_weights is not None:
        # broadcast weights across batch/edges
        w = dim_weights.to(diff.device).view(*([1] * (diff.dim() - 1)), -1)
        diff = diff * w

    if mask.any():
        vals = diff[mask]
        return vals.sum() if reduction == "sum" else vals.mean()
    else:
        return torch.zeros(1, device=pred.device).sum()  # scalar 0 on correct device


def _masked_sq(pred, target, threshold, dim_weights=None, reduction="sum"):
    """
    L2 with mask (|target| > threshold). Optional per-dimension weights.
    """
    if threshold is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        mask = target.abs() > threshold

    diff2 = (pred - target) ** 2  # [*, D]

    if dim_weights is not None:
        w = dim_weights.to(diff2.device).view(*([1] * (diff2.dim() - 1)), -1)
        diff2 = diff2 * w

    if mask.any():
        vals = diff2[mask]
        return vals.sum() if reduction == "sum" else vals.mean()
    else:
        return torch.zeros(1, device=pred.device).sum()


# ---------------- Node-only (backward compatible) ----------------

class L1_Loss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, node_out, node_y, accuracy_threshold):
        # Node-only L1 (masked), same behavior as your original (sum by default)
        return _masked_abs(node_out, node_y, accuracy_threshold, reduction=self.reduction)


class L2_Loss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, node_out, node_y, accuracy_threshold):
        return _masked_sq(node_out, node_y, accuracy_threshold, reduction=self.reduction)


# ---------------- Combined Node + Edge losses ----------------

class NodeEdgeL1Loss(nn.Module):
    """
    Combined masked L1 for nodes (2 dims: [dispX, dispY]) and edges (6 dims:
    [axial_shear, shear2, shear3, moment2, moment3, torsion]).
    You can weight node vs edge, and per-dimension within each head.
    """
    def __init__(self, node_weight=1.0, edge_weight=1.0, reduction="sum"):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.reduction = reduction

    def forward(
        self,
        node_out, node_y,
        edge_out=None, edge_y=None,
        node_thresh=1e-4, edge_thresh=1e-4,
        node_dim_weights=None,  # shape [2] or None
        edge_dim_weights=None   # shape [6] or None
    ):
        # Node part (required)
        node_loss = _masked_abs(
            node_out, node_y, node_thresh, dim_weights=node_dim_weights, reduction=self.reduction
        )

        # Edge part (optional)
        if (edge_out is not None) and (edge_y is not None):
            edge_loss = _masked_abs(
                edge_out, edge_y, edge_thresh, dim_weights=edge_dim_weights, reduction=self.reduction
            )
        else:
            edge_loss = torch.zeros(1, device=node_out.device).sum()

        return self.node_weight * node_loss + self.edge_weight * edge_loss


class NodeEdgeL2Loss(nn.Module):
    """Same as above but L2."""
    def __init__(self, node_weight=1.0, edge_weight=1.0, reduction="sum"):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.reduction = reduction

    def forward(
        self,
        node_out, node_y,
        edge_out=None, edge_y=None,
        node_thresh=1e-4, edge_thresh=1e-4,
        node_dim_weights=None,  # shape [2] or None
        edge_dim_weights=None   # shape [6] or None
    ):
        node_loss = _masked_sq(
            node_out, node_y, node_thresh, dim_weights=node_dim_weights, reduction=self.reduction
        )
        if (edge_out is not None) and (edge_y is not None):
            edge_loss = _masked_sq(
                edge_out, edge_y, edge_thresh, dim_weights=edge_dim_weights, reduction=self.reduction
            )
        else:
            edge_loss = torch.zeros(1, device=node_out.device).sum()

        return self.node_weight * node_loss + self.edge_weight * edge_loss
