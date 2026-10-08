"""Scale-invariant rank blending, logit combinations, and ensembling algorithms."""

from typing import Literal
import numpy as np
from scipy.stats import rankdata


def to_rank01(arr: np.ndarray) -> np.ndarray:
    """Computes percentile rank in [0, 1] for each column independently.

    Formula: R = (rank(P) - 1) / (N - 1)
    Eliminates calibration mismatch and temperature skew across different backbones.
    """
    n = arr.shape[0]
    if n <= 1:
        return np.full_like(arr, 0.5)

    ranks = np.zeros_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        r = rankdata(arr[:, col], method='average')
        ranks[:, col] = (r - 1.0) / (n - 1.0)
    return ranks


def to_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Clamps probabilities and maps to unconstrained log-odds space."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def from_logit(l: np.ndarray) -> np.ndarray:
    """Inverts log-odds to probabilities via sigmoid."""
    return 1.0 / (1.0 + np.exp(-l))


def blend_predictions(
    p_conv: np.ndarray,
    p_rad: np.ndarray,
    weight_conv: float = 0.65,
    weight_rad: float = 0.35,
    method: Literal["rank01", "logit", "linear"] = "rank01"
) -> np.ndarray:
    """Blends ConvNeXt and RadImageNet prediction matrices using the selected ensembling strategy.

    Parameters:
        p_conv: (N, C) probability matrix from ConvNeXt models.
        p_rad: (N, C) probability matrix from RadImageNet models.
        weight_conv: Float weight for ConvNeXt (default: 0.65).
        weight_rad: Float weight for RadImageNet (default: 0.35).
        method: 'rank01' (Recommended for ROC-AUC), 'logit', or 'linear'.

    Returns:
        (N, C) blended probability matrix strictly in [0.001, 0.999].
    """
    total_w = weight_conv + weight_rad
    w_c = weight_conv / total_w
    w_r = weight_rad / total_w

    if method.lower() == "rank01":
        if p_conv.shape[0] <= 1:
            # For single-sample evaluation, percentile rank across cohort is undefined (N-1=0).
            # Fall back gracefully to weighted probability combination.
            blended = w_c * p_conv + w_r * p_rad
        else:
            r_c = to_rank01(p_conv)
            r_r = to_rank01(p_rad)
            blended = w_c * r_c + w_r * r_r
    elif method.lower() == "logit":
        l_c = to_logit(p_conv)
        l_r = to_logit(p_rad)
        blended = from_logit(w_c * l_c + w_r * l_r)
    elif method.lower() == "linear":
        blended = w_c * p_conv + w_r * p_rad
    else:
        raise ValueError(f"Unknown blend method '{method}'. Choose from 'rank01', 'logit', 'linear'.")

    return np.clip(blended, 0.001, 0.999)
