"""Explainable AI (xAI) Engine for RSNA Knee Abnormality Detection.

Hierarchical 2-Tier Explanation Architecture:
  1. Tier 1 (Which Slice?): 1D volume depth attention profiling identifies the most
     diagnostically salient series slot (e.g., SAG_FS) and peak slice index.
  2. Tier 2 (Where in the Image?): 2D Grad-CAM backpropagation on that peak slice
     pinpoints the anatomical lesion/tear site.

Supports 4 CLI interaction modes via a single unified `--xai` flag:
  - `--xai` (no arg): explains all findings with predicted confidence > 0.70.
  - `--xai 0.6`: explains all findings with predicted confidence > 0.60.
  - `--xai top1`: explains the single highest-probability finding.
  - `--xai ACL`: explains the specified finding by name or abbreviation.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm

from src.constants import TARGET_COLS, CONVNEXT_SERIES_SLOTS


# -------------------------------------------------------------------------
# 1. Target Resolution
# -------------------------------------------------------------------------

def resolve_xai_targets(
    xai_spec: Optional[str],
    probs: np.ndarray,
    target_cols: List[str] = TARGET_COLS
) -> List[Dict[str, Any]]:
    """Resolves which finding(s) to explain based on user's `--xai` specification.

    Args:
        xai_spec: The value of the `--xai` CLI argument (or None if disabled).
        probs: 1D array of predicted probabilities of length len(target_cols).
        target_cols: List of target finding column names.

    Returns:
        List of dictionaries with keys:
            'index': int class index
            'name': str finding name
            'prob': float predicted probability
            'reason': str explanatory selection reason
    """
    if xai_spec is None:
        return []

    spec = str(xai_spec).strip()
    probs = np.asarray(probs, dtype=np.float32)

    # Mode 1: Top-1 (Dominant Finding / Most Votes)
    if spec.lower() in ["top1", "top-1", "highest", "max", "top"]:
        top_idx = int(np.argmax(probs))
        return [{
            "index": top_idx,
            "name": target_cols[top_idx],
            "prob": float(probs[top_idx]),
            "reason": f"Top-1 Highest Confidence ({probs[top_idx]:.1%})"
        }]

    # Mode 2: Numerical Confidence Threshold (e.g. '0.7', '0.6', or bare '--xai')
    try:
        threshold = float(spec)
        matches = [
            {
                "index": i,
                "name": target_cols[i],
                "prob": float(probs[i]),
                "reason": f"Confidence > {threshold:.2f} ({probs[i]:.1%})"
            }
            for i in range(len(target_cols))
            if probs[i] >= threshold
        ]
        # Sort matches by probability descending
        matches.sort(key=lambda m: m["prob"], reverse=True)

        # Fallback to Top-1 if no finding crossed the threshold
        if not matches:
            top_idx = int(np.argmax(probs))
            return [{
                "index": top_idx,
                "name": target_cols[top_idx],
                "prob": float(probs[top_idx]),
                "reason": f"Highest Candidate (Below {threshold:.2f} threshold: {probs[top_idx]:.1%})"
            }]
        return matches
    except ValueError:
        pass  # Not a float, proceed to string/label matching

    # Mode 3: Target Name or Abbreviation Matching (e.g. 'ACL', 'Meniscus', 'Effusion')
    spec_clean = spec.lower().replace("_", " ").replace("-", " ")
    exact_matches = []
    sub_matches = []

    for i, col in enumerate(target_cols):
        col_clean = col.lower().replace("_", " ").replace("-", " ")
        if spec_clean == col_clean:
            exact_matches.append(i)
        elif spec_clean in col_clean or col_clean in spec_clean:
            sub_matches.append(i)

    chosen_indices = exact_matches if exact_matches else sub_matches

    if chosen_indices:
        return [
            {
                "index": i,
                "name": target_cols[i],
                "prob": float(probs[i]),
                "reason": f"Requested Target '{spec}' ({probs[i]:.1%})"
            }
            for i in chosen_indices
        ]

    # Mode 4: Unrecognized target label fallback
    top_idx = int(np.argmax(probs))
    return [{
        "index": top_idx,
        "name": target_cols[top_idx],
        "prob": float(probs[top_idx]),
        "reason": f"Unknown Target '{spec}'; Fallback to Top-1 ({probs[top_idx]:.1%})"
    }]


# -------------------------------------------------------------------------
# 2. Tier 1: Slice-Level Depth Attention Extraction
# -------------------------------------------------------------------------

def extract_slice_attention(
    attn_weights: torch.Tensor,
    mask: np.ndarray,
    target_idx: int,
    slots: List[str] = CONVNEXT_SERIES_SLOTS
) -> Tuple[int, int, str, np.ndarray]:
    """Extracts 1D slice attention weights and identifies the peak diagnostic slice.

    Args:
        attn_weights: Tensor of shape (1, total_slices, NUM_CLASSES) from model forward pass.
        mask: 1D boolean array indicating valid slots (e.g. shape (6,)).
        target_idx: Target finding class index (0 to 11).
        slots: Names of canonical series slots.

    Returns:
        peak_slot_idx: Index of the winning series slot.
        peak_slice_idx: Index of peak slice within that winning slot.
        peak_slot_name: Canonical slot name (e.g. 'SAG_FS').
        slot_attn_curve: 1D numpy array of attention weights across slices of winning slot.
    """
    if attn_weights.dim() == 3:
        w = attn_weights[0, :, target_idx].detach().cpu().numpy()  # (total_slices,)
    else:
        w = attn_weights[:, target_idx].detach().cpu().numpy()

    n_slots = len(slots)
    k_slices = len(w) // n_slots
    w_grid = w.reshape(n_slots, k_slices).copy()

    # Mask out absent slots
    valid_mask = np.asarray(mask, dtype=bool)
    w_grid[~valid_mask] = -1.0

    # Locate peak slot and slice
    peak_flat = np.argmax(w_grid)
    peak_slot_idx, peak_slice_idx = np.unravel_index(peak_flat, w_grid.shape)
    peak_slot_name = slots[peak_slot_idx]
    slot_attn_curve = np.maximum(0.0, w_grid[peak_slot_idx])

    # Normalize curve for plotting
    curve_sum = slot_attn_curve.sum()
    if curve_sum > 1e-6:
        slot_attn_curve = slot_attn_curve / curve_sum

    return int(peak_slot_idx), int(peak_slice_idx), peak_slot_name, slot_attn_curve


# -------------------------------------------------------------------------
# 3. Tier 2: 2D Spatial Grad-CAM on Peak Slice
# -------------------------------------------------------------------------

def _find_gradcam_target_layer(backbone: nn.Module) -> nn.Module:
    """Finds the optimal penultimate feature layer for Grad-CAM."""
    # ConvNeXt architectures
    if hasattr(backbone, "stages") and len(backbone.stages) > 0:
        last_stage = backbone.stages[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1]
        return last_stage
    # ResNet architectures
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    # General fallback: last Conv2d
    target = None
    for _, mod in backbone.named_modules():
        if isinstance(mod, nn.Conv2d):
            target = mod
    return target if target is not None else backbone


def compute_gradcam_slab(
    model: nn.Module,
    slab: torch.Tensor,
    target_idx: int,
    device: torch.device
) -> np.ndarray:
    """Computes 2D Grad-CAM heatmap for a single 2.5D tri-slice slab.

    Args:
        model: Trained ConvNeXt6View model.
        slab: Tensor of shape (1, 3, H, W) representing [s_{t-1}, s_t, s_{t+1}].
        target_idx: Class index to backpropagate from.
        device: Active PyTorch device.

    Returns:
        2D numpy array of shape (H, W) with heatmap values normalized in [0, 1].
    """
    slab = slab.to(device)
    if slab.dim() == 3:
        slab = slab.unsqueeze(0)

    # Locate target layer in backbone
    backbone = getattr(model, "backbone", model)
    target_layer = _find_gradcam_target_layer(backbone)

    activations = []
    gradients = []

    def forward_hook(mod, inp, out):
        activations.append(out)

    def backward_hook(mod, grad_inp, grad_out):
        gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Enable gradient computation locally
        with torch.enable_grad():
            slab_var = slab.clone().detach().requires_grad_(True)
            feat = backbone(slab_var)  # (1, in_features)

            # Compute effective class direction vector through projection & classifier head
            if hasattr(model, "proj") and hasattr(model, "cls_w"):
                proj_w = model.proj[1].weight  # (dim, in_features)
                cls_w = model.cls_w[target_idx]  # (dim,)
                # Direction vector in backbone feature space
                direction = torch.matmul(cls_w, proj_w)  # (in_features,)
                score = (feat * direction.unsqueeze(0)).sum()
            else:
                score = feat.sum()

            backbone.zero_grad()
            if hasattr(model, "zero_grad"):
                model.zero_grad()
            score.backward()

        if not activations or not gradients:
            # Fallback uniform heatmap if hooking was skipped
            return np.ones((slab.shape[-2], slab.shape[-1]), dtype=np.float32) * 0.5

        act = activations[0].detach()  # (1, C, h, w)
        grad = gradients[0].detach()   # (1, C, h, w)

        # Global average pool gradients along spatial dimensions to get channel weights
        alpha = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((alpha * act).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        # Upsample heatmap to original slice resolution
        cam = F.interpolate(cam, size=(slab.shape[-2], slab.shape[-1]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    finally:
        h_fwd.remove()
        h_bwd.remove()


# -------------------------------------------------------------------------
# 4. Visualization & Artifact Rendering
# -------------------------------------------------------------------------

def render_xai_card(
    raw_slice: np.ndarray,
    heatmap: np.ndarray,
    attn_curve: np.ndarray,
    peak_slice_idx: int,
    slot_name: str,
    finding_name: str,
    prob: float,
    reason: str,
    output_path: Path,
    study_uid: str = "Unknown"
) -> Path:
    """Renders a publication-ready 3-panel XAI visual explanation card.

    Panels:
        1. Tier 1: 1D Depth Slice Attention Profile (with peak slice marked).
        2. Key Diagnostic Slice (Raw MRI grayscale image).
        3. Tier 2: Anatomical Lesion Heatmap Overlay (MRI + Grad-CAM).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(15, 5), facecolor="#1e1e24")
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # ---------------- Panel 1: Slice Attention Profile ----------------
    ax1 = axes[0]
    ax1.set_facecolor("#282830")
    slices = np.arange(len(attn_curve))
    ax1.plot(slices, attn_curve, color="#00bcd4", linewidth=2.5, marker="o", markersize=4, label="Attention Weight")
    ax1.axvline(peak_slice_idx, color="#ff5252", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.plot(peak_slice_idx, attn_curve[peak_slice_idx], color="#ff5252", marker="*", markersize=14, label=f"Peak Slice: {peak_slice_idx}")

    ax1.set_title(f"Tier 1: Depth Attention Profile\nSeries Slot: {slot_name}", color="white", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Slice Index in Volume", color="#cccccc", fontsize=9)
    ax1.set_ylabel("Attention Score", color="#cccccc", fontsize=9)
    ax1.tick_params(colors="#cccccc")
    for spine in ax1.spines.values():
        spine.set_color("#444444")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#888888")
    ax1.legend(loc="upper right", facecolor="#1e1e24", edgecolor="#444444", labelcolor="white", fontsize=8)

    # ---------------- Panel 2: Key Diagnostic Slice ----------------
    ax2 = axes[1]
    ax2.set_facecolor("#000000")
    norm_slice = raw_slice.astype(np.float32)
    if norm_slice.max() > 1.0:
        norm_slice = norm_slice / 255.0
    ax2.imshow(norm_slice, cmap="gray", origin="upper")
    ax2.set_title(f"Diagnostic Cut: Slice {peak_slice_idx}\n({slot_name})", color="white", fontsize=11, fontweight="bold", pad=10)
    ax2.axis("off")

    # ---------------- Panel 3: Lesion Heatmap Overlay ----------------
    ax3 = axes[2]
    ax3.set_facecolor("#000000")
    ax3.imshow(norm_slice, cmap="gray", origin="upper")
    # Colormap overlay with transparency
    im_cam = ax3.imshow(heatmap, cmap="jet", alpha=0.45, origin="upper")
    ax3.set_title(f"Tier 2: Lesion Grad-CAM Overlay\n{finding_name}: {prob:.1%}", color="#ffeb3b", fontsize=11, fontweight="bold", pad=10)
    ax3.axis("off")

    # Overall title
    fig.suptitle(
        f"RSNA Knee MRI Explainable AI (xAI) | Study: {study_uid[:18]}...\n"
        f"Target: {finding_name} | Predicted Probability: {prob:.1%} | Reason: {reason}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")

    return output_path


# -------------------------------------------------------------------------
# 5. Advanced Multi-View xAI: Depth Profiles & Top-3 Diagnostic Cuts
# -------------------------------------------------------------------------

def extract_multiview_slice_attention(
    attn_weights: torch.Tensor,
    mask: np.ndarray,
    target_idx: int,
    slots: List[str] = CONVNEXT_SERIES_SLOTS,
    top_k: int = 3
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Extracts depth attention curves for all active series slots and identifies top-k cuts globally."""
    if attn_weights.dim() == 3:
        w = attn_weights[0, :, target_idx].detach().cpu().numpy()
    else:
        w = attn_weights[:, target_idx].detach().cpu().numpy()

    n_slots = len(slots)
    k_slices = len(w) // n_slots
    w_grid = w.reshape(n_slots, k_slices).copy()

    valid_mask = np.asarray(mask, dtype=bool)
    slot_curves = {}
    all_candidates = []

    for s_idx, s_name in enumerate(slots):
        if valid_mask[s_idx]:
            curve = np.maximum(0.0, w_grid[s_idx]).copy()
            slot_curves[s_name] = curve
            for sl_idx in range(k_slices):
                all_candidates.append({
                    "slot_idx": s_idx,
                    "slot_name": s_name,
                    "slice_idx": sl_idx,
                    "score": float(curve[sl_idx])
                })

    all_candidates.sort(key=lambda c: c["score"], reverse=True)

    top_cuts = []
    seen = set()
    for cand in all_candidates:
        key = (cand["slot_name"], cand["slice_idx"])
        if key not in seen:
            seen.add(key)
            cand_copy = cand.copy()
            cand_copy["rank"] = len(top_cuts) + 1
            top_cuts.append(cand_copy)
            if len(top_cuts) >= top_k:
                break

    return slot_curves, top_cuts


def render_multiview_attention_curve(
    slot_curves: Dict[str, np.ndarray],
    top_cuts: List[Dict[str, Any]],
    finding_name: str,
    prob: float,
    output_path: Path,
    study_uid: str = "Unknown"
) -> Path:
    """Renders Plot 1: Multi-View Attention Profile with distinct curves per view slot and top-3 badges."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(14, 5.5), facecolor="#141416")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1f1f23")

    palette = {
        "SAG_N": "#00e5ff",
        "SAG_FS": "#f50057",
        "COR_N": "#ffd600",
        "COR_FS": "#00e676",
        "AX_N": "#ff9100",
        "AX_FS": "#d500f9"
    }

    for slot_name, curve in slot_curves.items():
        color = palette.get(slot_name, "#38bdf8")
        slices = np.arange(len(curve))
        ax.plot(
            slices, curve, color=color, linewidth=2.5, alpha=0.9,
            marker="o", markersize=4.5, label=f"{slot_name} (Peak: {curve.max():.3f})"
        )

    rank_styles = {
        1: {"color": "#ffd700", "marker": "*", "size": 17, "label": "Rank 1 (Gold)"},
        2: {"color": "#e0e0e0", "marker": "*", "size": 14, "label": "Rank 2 (Silver)"},
        3: {"color": "#cd7f32", "marker": "*", "size": 12, "label": "Rank 3 (Bronze)"}
    }

    for cut in top_cuts:
        r = cut.get("rank", 1)
        style = rank_styles.get(r, rank_styles[3])
        x_pos = cut["slice_idx"]
        y_pos = cut["score"]
        ax.plot(x_pos, y_pos, color=style["color"], marker=style["marker"], markersize=style["size"], zorder=5)
        y_offset = 0.012 if r % 2 == 1 else -0.018
        ax.annotate(
            f"#{r}: {cut['slot_name']} S{x_pos}\n({y_pos:.3f})",
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos + y_offset),
            arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.2),
            color=style["color"],
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#141416", alpha=0.85)
        )

    ax.set_title(
        f"Tier 1: Multi-View Slice Depth Attention Profile | Target: {finding_name} ({prob:.1%})\n"
        f"Top 3 Diagnostic Slices Highlighted Across Anatomical Slots | Study: {study_uid[:24]}...",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Slice Index Along Volume Depth", color="#cccccc", fontsize=10)
    ax.set_ylabel(f"Attention Weight ({finding_name})", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.grid(True, linestyle=":", alpha=0.35, color="#71717a")
    ax.legend(loc="upper right", facecolor="#141416", edgecolor="#3f3f46", labelcolor="white", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    return output_path


def render_top_slices_gradcam_grid(
    model: nn.Module,
    tx: torch.Tensor,
    slabs: np.ndarray,
    top_cuts: List[Dict[str, Any]],
    target_idx: int,
    finding_name: str,
    prob: float,
    device: torch.device,
    output_path: Path,
    study_uid: str = "Unknown"
) -> Path:
    """Renders Plot 2: Side-by-side Diagnostic Cuts & Grad-CAM Heatmaps for the top-3 cuts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cuts = min(3, len(top_cuts))
    if n_cuts == 0:
        return output_path

    fig = Figure(figsize=(5.2 * n_cuts, 7.0), facecolor="#141416")
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(2, n_cuts)
    if n_cuts == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    rank_badges = {1: "#1 (Primary Cut)", 2: "#2 (Second Best)", 3: "#3 (Third Best)"}

    for col_idx in range(n_cuts):
        cut = top_cuts[col_idx]
        s_idx = cut["slot_idx"]
        sl_idx = cut["slice_idx"]
        slot_name = cut["slot_name"]
        score = cut["score"]
        rank = cut.get("rank", col_idx + 1)

        raw_slice = slabs[s_idx, sl_idx, 1]
        norm_slice = raw_slice.astype(np.float32)
        if norm_slice.max() > 1.0:
            norm_slice = norm_slice / 255.0

        slab_tensor = tx[:, s_idx, sl_idx:sl_idx+1, :, :, :].squeeze(1)
        cam = compute_gradcam_slab(model, slab_tensor, target_idx, device)

        # Row 1: Raw Diagnostic Cut
        ax_top = axes[0, col_idx]
        ax_top.set_facecolor("#000000")
        ax_top.imshow(norm_slice, cmap="gray", origin="upper")
        ax_top.set_title(
            f"Rank {rank_badges.get(rank, f'#{rank}')}\n{slot_name} — Slice {sl_idx}\n(Attn Score: {score:.3f})",
            color="#38bdf8", fontsize=10.5, fontweight="bold", pad=8
        )
        ax_top.axis("off")

        # Row 2: Grad-CAM Overlay
        ax_bot = axes[1, col_idx]
        ax_bot.set_facecolor("#000000")
        ax_bot.imshow(norm_slice, cmap="gray", origin="upper")
        ax_bot.imshow(cam, cmap="jet", alpha=0.45, origin="upper")
        ax_bot.set_title(
            f"Lesion Grad-CAM Overlay\n{finding_name}: {prob:.1%}",
            color="#ffd700", fontsize=10.5, fontweight="bold", pad=8
        )
        ax_bot.axis("off")

    fig.suptitle(
        f"Tier 2: Grad-CAM Lesion Overlays Across Top Views | Target: {finding_name} ({prob:.1%})\nStudy: {study_uid[:28]}...",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    return output_path


def render_finding_xai_duo(
    attn_weights: torch.Tensor,
    mask: np.ndarray,
    slabs: np.ndarray,
    tx: torch.Tensor,
    model: nn.Module,
    target_idx: int,
    finding_name: str,
    score: float,
    risk_tier: str,
    device: torch.device,
    series_slots: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    study_uid: str = ""
) -> Optional[Path]:
    """Renders a 2-panel figure: Left side has multi-view depth attention curves,
    Right side has the single peak diagnostic slice with Grad-CAM lesion heatmap overlay.
    """
    from src.constants import CONVNEXT_SERIES_SLOTS
    series_slots = series_slots if series_slots is not None else CONVNEXT_SERIES_SLOTS

    # 1. Extract multi-view curves & identify Top-1 peak slice
    slot_curves, top_cuts = extract_multiview_slice_attention(
        attn_weights, mask, target_idx, series_slots, top_k=1
    )

    if not top_cuts:
        return None

    peak_cut = top_cuts[0]
    peak_slot = peak_cut["slot_idx"]
    peak_slice = peak_cut["slice_idx"]
    slot_name = peak_cut["slot_name"]
    attn_score = peak_cut["score"]

    # 2. Extract peak raw slice and compute Grad-CAM
    raw_slice = slabs[peak_slot, peak_slice, 1]
    norm_slice = raw_slice.astype(np.float32)
    if norm_slice.max() > 1.0:
        norm_slice = norm_slice / 255.0

    slab_tensor = tx[:, peak_slot, peak_slice:peak_slice+1, :, :, :].squeeze(1)
    cam = compute_gradcam_slab(model, slab_tensor, target_idx, device)

    # 3. Create 2-panel figure: Left = Curves, Right = Heatmap
    fig = Figure(figsize=(14.0, 5.5), facecolor="#18181b")
    canvas = FigureCanvasAgg(fig)
    ax_curve, ax_cam = fig.subplots(1, 2)

    SLOT_COLORS = {
        "SAG_N": "#38bdf8",
        "SAG_FS": "#4ade80",
        "COR_N": "#f59e0b",
        "COR_FS": "#fb923c",
        "AX_N": "#a855f7",
        "AX_FS": "#f43f5e"
    }

    # Left Panel: Multi-view Attention Curves
    ax_curve.set_facecolor("#09090b")
    for s_name, curve in slot_curves.items():
        color = SLOT_COLORS.get(s_name, "#94a3b8")
        ax_curve.plot(curve, label=s_name, color=color, lw=2.2, alpha=0.9)

    # Mark peak slice on winning curve
    peak_val = slot_curves[slot_name][peak_slice] if slot_name in slot_curves else 0.0
    ax_curve.scatter([peak_slice], [peak_val], color="#f43f5e", s=180, zorder=5, marker="*", edgecolor="white", lw=1.5)
    ax_curve.axvline(peak_slice, color="#f43f5e", ls="--", alpha=0.6, lw=1.5)

    ax_curve.set_title(
        f"Multi-View Attention Profile\nPeak Diagnostic Cut -> {slot_name} Slice {peak_slice:02d}",
        color="#e2e8f0", fontsize=11, fontweight="bold", pad=8
    )
    ax_curve.set_xlabel("Slice Index in Volume", color="#cbd5e1", fontsize=10, labelpad=6)
    ax_curve.set_ylabel("Attention Weight", color="#cbd5e1", fontsize=10, labelpad=6)
    ax_curve.tick_params(colors="#94a3b8")
    ax_curve.grid(True, color="#27272a", alpha=0.7)
    ax_curve.legend(loc="upper right", facecolor="#18181b", edgecolor="#3f3f46", labelcolor="#e2e8f0", fontsize=9)

    # Right Panel: Single Diagnostic Cut with Grad-CAM Overlay
    ax_cam.set_facecolor("#09090b")
    ax_cam.imshow(norm_slice, cmap="gray", origin="upper")
    ax_cam.imshow(cam, cmap="jet", alpha=0.45, origin="upper")
    ax_cam.set_title(
        f"Diagnostic Cut: {slot_name} Slice {peak_slice:02d}\nLesion Grad-CAM Overlay",
        color="#e2e8f0", fontsize=11, fontweight="bold", pad=8
    )
    ax_cam.axis("off")

    # title shoudl be 

    fig.suptitle(
        f"{finding_name} | Score: {score:.1%} | {risk_tier.upper()}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        return output_path
    return None

