"""Loss functions for RSNA Knee Abnormality Detection.

Supports:
  1. SampleWeightedBCEWithLogitsLoss: Standard Binary Cross-Entropy with per-sample loss weighting
     (used for ConvNeXt continuous soft teacher supervision and 3x Gold 58 benchmark weighting).
  2. SampleWeightedFocalLoss: Multi-label Focal Loss with per-sample weighting
     (used for RadImageNet to combat severe class imbalance in rare findings such as fractures and contusions).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleWeightedBCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy with Logits and per-sample continuous loss weighting.

    Loss formula:
        L_i = BCE(logits_i, targets_i)  # (C,)
        Loss = (sum_i w_i * mean_c(L_{ic})) / (sum_i w_i)
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes sample-weighted BCE loss.

        Args:
            logits: (B, C) raw unnormalized model logits.
            targets: (B, C) continuous or binary target labels in [0, 1].
            sample_weights: (B,) optional per-sample loss multipliers.

        Returns:
            Scalar loss tensor.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )  # (B, C)

        if sample_weights is not None:
            w = sample_weights.view(-1, 1).to(bce.dtype)
            weighted_bce = bce * w
            if self.reduction == "mean":
                denom = w.sum().clamp_min(1.0) * logits.shape[-1]
                return weighted_bce.sum() / denom
            elif self.reduction == "sum":
                return weighted_bce.sum()
            return weighted_bce
        else:
            if self.reduction == "mean":
                return bce.mean()
            elif self.reduction == "sum":
                return bce.sum()
            return bce


class SampleWeightedFocalLoss(nn.Module):
    """Multi-Label Focal Loss with Logits and per-sample continuous loss weighting.

    Focal Loss dynamically suppresses easy negative background predictions (p_t ~ 1.0)
    by multiplying BCE by (1 - p_t)^gamma, directing all optimization gradients toward
    hard ambiguous tears and rare clinical pathologies (e.g. Fractures, Contusions, Baker's).

    Loss formula:
        p_t = exp(-BCE(logits, targets))
        FL = (1 - p_t)^gamma * BCE(logits, targets)
        Loss = (sum_i w_i * sum_c(FL_{ic})) / (sum_i w_i * C)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes sample-weighted Focal loss.

        Args:
            logits: (B, C) raw unnormalized model logits.
            targets: (B, C) continuous or binary target labels in [0, 1].
            sample_weights: (B,) optional per-sample loss multipliers.

        Returns:
            Scalar loss tensor.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )  # (B, C)

        # Numerically stable p_t = exp(-bce)
        pt = torch.exp(-bce)

        if self.gamma > 0.0:
            focal_modulator = (1.0 - pt) ** self.gamma
            focal_loss = focal_modulator * bce
        else:
            focal_loss = bce

        if sample_weights is not None:
            w = sample_weights.view(-1, 1).to(focal_loss.dtype)
            weighted_focal = focal_loss * w
            if self.reduction == "mean":
                denom = w.sum().clamp_min(1.0) * logits.shape[-1]
                return weighted_focal.sum() / denom
            elif self.reduction == "sum":
                return weighted_focal.sum()
            return weighted_focal
        else:
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            return focal_loss


def get_loss_function(
    name: str = "bce",
    gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> nn.Module:
    """Factory helper to retrieve the appropriate loss module.

    Args:
        name: 'bce' or 'focal'
        gamma: focusing parameter for focal loss (default: 2.0)
        pos_weight: optional positive class weighting tensor
        reduction: 'mean' or 'sum'

    Returns:
        Instantiated nn.Module callable as loss(logits, targets, sample_weights).
    """
    clean_name = str(name).lower().strip()
    if clean_name in ("focal", "focal_loss", "focalloss"):
        return SampleWeightedFocalLoss(gamma=gamma, pos_weight=pos_weight, reduction=reduction)
    elif clean_name in ("bce", "bcewithlogits", "bce_with_logits", "binary_cross_entropy"):
        return SampleWeightedBCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function '{name}'. Choose 'bce' or 'focal'.")
