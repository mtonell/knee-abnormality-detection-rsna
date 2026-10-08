#!/usr/bin/env python3
"""Unified End-to-End Test Inference CLI for RSNA Knee Abnormality Detection.

Executes test inference directly on raw DICOM series without requiring offline preprocessing caches.

Usage:
  # Flagship 65/35 Rank01 Hybrid Blend on sample examinations:
  python infer.py --data_dir data/samples --output submission.csv

  # Standalone ConvNeXt inference:
  python infer.py --model convnext --data_dir data/samples --output submission_conv.csv

  # Standalone RadImageNet inference:
  python infer.py --model radimagenet --data_dir data/samples --output submission_rad.csv
"""

import os
import gc
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

# Ensure repo root is on python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    TARGET_COLS,
    NUM_CLASSES,
    CONVNEXT_SERIES_SLOTS,
    RADIMAGENET_SERIES_SLOTS,
    DEFAULT_IMAGE_SIZE
)
from src.dicom import extract_study_dicom_volumes, build_2p5d_slabs, read_dicom_pixels, percentile_window_uint8
from src.geometry import crop_and_pad_square, apply_laterality_normalization
from src.models.convnext import ConvNeXt6View, load_convnext_checkpoint
from src.models.radimagenet import RadImageNetEncoder, FoundationQueryHead, load_radimagenet_bundle
from src.ensembling import blend_predictions
from src.xai import (
    resolve_xai_targets,
    extract_slice_attention,
    compute_gradcam_slab,
    render_xai_card
)

# Model-specific slice retention defaults
CONVNEXT_DEFAULT_BUDGET = {"SAG": 20, "COR": 20, "AX": 16}
CONVNEXT_DEFAULT_BAND = (0.04, 0.96)
RADIMAGENET_DEFAULT_BUDGET = {"SAG": 8, "COR": 8, "AX": 8}
RADIMAGENET_DEFAULT_BAND = (0.12, 0.88)


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA Knee Test Inference & Ensembling CLI")
    parser.add_argument("--data_dir", type=str, default="data/samples",
                        help="Path to directory containing study UID folders with DICOM files.")
    parser.add_argument("--series_csv", type=str, default=None,
                        help="Optional path to series metadata CSV (auto-detected if present).")
    parser.add_argument("--model", type=str, default="blend", choices=["blend", "convnext", "radimagenet"],
                        help="Model to run: 'blend' (70/30 Hybrid), 'convnext', or 'radimagenet'.")
    parser.add_argument("--blend_method", type=str, default="rank01", choices=["rank01", "logit", "linear"],
                        help="Ensemble strategy for hybrid blend (default: 'rank01' SOTA).")
    parser.add_argument("--weight_conv", type=float, default=0.7,
                        help="Ensemble blend weight for ConvNeXt (default: 0.65).")
    parser.add_argument("--weight_rad", type=float, default=0.3,
                        help="Ensemble blend weight for RadImageNet (default: 0.35).")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output path for final submission CSV.")
    parser.add_argument("--conv_ckpt", type=str, default="checkpoints/convnext_full_distilled_best.pt",
                        help="Path to ConvNeXt checkpoint file or directory.")
    parser.add_argument("--rad_ckpt", type=str, default="checkpoints/v52_e11_heads_v1.pt",
                        help="Path to RadImageNet query heads bundle file or directory.")
    parser.add_argument("--resnet_ckpt", type=str, default="checkpoints/ResNet50.pt",
                        help="Path to RadImageNet ResNet-50 backbone checkpoint.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'cpu' (default: autodetect).")
    parser.add_argument("--xai", nargs="?", const="0.7", default=None,
                        help="Generate Explainable AI (xAI) depth attention & Grad-CAM overlays. "
                             "Usage: '--xai' (>0.7), '--xai 0.6' (>0.6), '--xai top1', or '--xai ACL'.")
    parser.add_argument("--xai_out_dir", type=str, default="outputs/xai",
                        help="Output directory to save visual explanation cards (default: 'outputs/xai').")
    return parser.parse_args()


RESERVED_DIR_NAMES = {
    "test_series", "train_series", "series", "images", "samples",
    "checkpoints", "cache", "outputs", "notebooks", "src", "configs",
    "data", "assets", "models"
}


def discover_studies(data_dir: Path, series_df: Optional[pd.DataFrame] = None) -> List[str]:
    """Finds genuine study UIDs, avoiding reserved dataset directory names."""
    # Priority 1: If series metadata is already provided, use its StudyInstanceUID column
    if series_df is not None and not series_df.empty and "StudyInstanceUID" in series_df.columns:
        valid_uids = [str(u) for u in series_df["StudyInstanceUID"].dropna().unique() if str(u) not in RESERVED_DIR_NAMES]
        if valid_uids:
            return sorted(valid_uids)

    # Priority 2: Check for test.csv or train.csv in data_dir or its parent
    for cand_csv in [data_dir / "test.csv", data_dir.parent / "test.csv", data_dir / "train.csv"]:
        if cand_csv.exists():
            try:
                df = pd.read_csv(cand_csv)
                if "StudyInstanceUID" in df.columns:
                    uids = [str(u) for u in df["StudyInstanceUID"].dropna().unique() if str(u) not in RESERVED_DIR_NAMES]
                    if uids:
                        return sorted(uids)
            except Exception:
                pass

    # Priority 3: Scan directories directly, ignoring reserved system folders
    search_dirs = [data_dir]
    for sub_name in ["test_series", "train_series", "samples"]:
        sub_path = data_dir / sub_name
        if sub_path.exists() and sub_path.is_dir():
            search_dirs.append(sub_path)

    studies = []
    for sdir in search_dirs:
        for d in sdir.iterdir():
            if d.is_dir() and not d.name.startswith(".") and d.name not in RESERVED_DIR_NAMES:
                # Confirm it contains subdirectories or dicom files
                if any(d.glob("*.dcm")) or any(d.glob("*/*.dcm")):
                    studies.append(d.name)

    if not studies:
        # Fallback recursive search
        for sub in data_dir.rglob("*.dcm"):
            parent_name = sub.parent.name
            grandparent_name = sub.parent.parent.name if sub.parent.parent != data_dir else None
            candidate = grandparent_name if grandparent_name and grandparent_name not in RESERVED_DIR_NAMES else parent_name
            if candidate not in RESERVED_DIR_NAMES and candidate not in studies:
                studies.append(candidate)

    return sorted(list(set(studies)))


def auto_build_series_metadata(data_dir: Path, studies: List[str]) -> pd.DataFrame:
    """Inspects DICOM headers on the fly if no series.csv was provided."""
    import pydicom
    rows = []
    print("Extracting series metadata directly from DICOM headers...")
    for st in studies:
        st_dir = data_dir / st
        if not st_dir.exists():
            continue
        for s_dir in st_dir.iterdir():
            if s_dir.is_dir():
                dcm_files = list(s_dir.glob("*.dcm"))
                if dcm_files:
                    mid = dcm_files[len(dcm_files) // 2]
                    try:
                        ds = pydicom.dcmread(mid, stop_before_pixels=True)
                        desc = str(getattr(ds, "SeriesDescription", "") or "").lower()
                        plane = "Sagittal" if "sag" in desc else ("Coronal" if "cor" in desc else "Axial")
                        fs = 1 if any(t in desc for t in ["fs", "fat", "stir", "spair"]) else 0
                        rows.append({
                            "StudyInstanceUID": st,
                            "SeriesInstanceUID": s_dir.name,
                            "Anatomical_Plane": plane,
                            "Fat_Suppression": fs
                        })
                    except Exception:
                        pass
    return pd.DataFrame(rows)


def run_convnext_inference(
    studies: List[str],
    series_df: pd.DataFrame,
    data_dir: Path,
    ckpt_path: Path,
    device: torch.device,
    xai_spec: Optional[str] = None,
    xai_out_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Executes ConvNeXt-Nano 6-view Transformer test inference and optional xAI."""
    print(f"\n--- [1/2] Executing ConvNeXt-Nano Inference ({len(studies)} studies) ---")
    if xai_spec is not None:
        print(f"  [xAI Active] Spec: '{xai_spec}' | Saving visual cards to: {xai_out_dir}")
    if not ckpt_path.exists():
        print(f"[!] Warning: ConvNeXt checkpoint not found at {ckpt_path}. Using random initialized weights for demonstration.")
        model = ConvNeXt6View(num_classes=NUM_CLASSES).to(device).eval()
    else:
        model = load_convnext_checkpoint(ckpt_path, device)
        print(f"[+] Loaded ConvNeXt Checkpoint: {ckpt_path.name}")

    preds = []
    for st in tqdm(studies, desc="ConvNeXt Slicing & Inference"):
        st_series = series_df[series_df["StudyInstanceUID"] == st] if not series_df.empty else pd.DataFrame()
        vols, mask = extract_study_dicom_volumes(
            st, st_series, data_dir,
            slots=CONVNEXT_SERIES_SLOTS,
            slice_budget=CONVNEXT_DEFAULT_BUDGET,
            depth_band=CONVNEXT_DEFAULT_BAND
        )

        # Build 2.5D tri-slice slabs
        k = max(CONVNEXT_DEFAULT_BUDGET.values())
        slabs, pos = build_2p5d_slabs(vols, mask, k_budget=k, depth_band=CONVNEXT_DEFAULT_BAND)

        tx = torch.from_numpy(slabs).unsqueeze(0).to(device).float().div_(255.0)
        tm = torch.from_numpy(mask).unsqueeze(0).to(device)
        tp = torch.from_numpy(pos).unsqueeze(0).to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if xai_spec is not None:
                logits, attn_weights = model(tx, tm, tp, return_attention=True)
            else:
                with torch.no_grad():
                    logits = model(tx, tm, tp)
            prob = torch.sigmoid(logits).float().cpu().detach().numpy()[0]
        preds.append(prob)

        # Execute xAI if requested
        if xai_spec is not None:
            targets = resolve_xai_targets(xai_spec, prob, TARGET_COLS)
            if targets and xai_out_dir is not None:
                xai_out_dir.mkdir(parents=True, exist_ok=True)
                for t in targets:
                    c_idx = t["index"]
                    peak_slot, peak_slice, slot_name, attn_curve = extract_slice_attention(
                        attn_weights, mask, c_idx, CONVNEXT_SERIES_SLOTS
                    )
                    raw_slice = slabs[peak_slot, peak_slice, 1]
                    slab_tensor = tx[:, peak_slot, peak_slice:peak_slice+1, :, :, :].squeeze(1)
                    cam = compute_gradcam_slab(model, slab_tensor, c_idx, device)
                    finding_slug = t["name"].lower().replace(" ", "_").replace("'", "")
                    card_path = xai_out_dir / f"{st}_{finding_slug}_xai.png"
                    render_xai_card(
                        raw_slice=raw_slice,
                        heatmap=cam,
                        attn_curve=attn_curve,
                        peak_slice_idx=peak_slice,
                        slot_name=slot_name,
                        finding_name=t["name"],
                        prob=t["prob"],
                        reason=t["reason"],
                        output_path=card_path,
                        study_uid=st
                    )
                    print(f"  [xAI] {st[:14]}... | {t['name']}: {t['prob']:.1%} | {slot_name} Slice {peak_slice}/{k} -> {card_path.name}")

    df_out = pd.DataFrame(preds, columns=TARGET_COLS)
    df_out.insert(0, "StudyInstanceUID", studies)
    return df_out


def run_radimagenet_inference(
    studies: List[str],
    series_df: pd.DataFrame,
    data_dir: Path,
    heads_path: Path,
    resnet_path: Path,
    device: torch.device
) -> pd.DataFrame:
    """Executes RadImageNet ResNet-50 + Query Attention Head inference."""
    print(f"\n--- [2/2] Executing RadImageNet Inference ({len(studies)} studies) ---")
    if not resnet_path.exists():
        print(f"[!] Warning: ResNet50.pt not found at {resnet_path}. Using torchvision standard ResNet-50 weights.")
        from torchvision.models import resnet50
        base = resnet50(weights=None)
        encoder = nn.Sequential(*list(base.children())[:-2], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()).to(device).eval()
    else:
        encoder = RadImageNetEncoder(resnet_path).to(device)
        print(f"[+] Loaded RadImageNet Backbone: {resnet_path.name}")

    if not heads_path.exists():
        print(f"[!] Warning: Query head checkpoint not found at {heads_path}. Using initialized head for demonstration.")
        heads = [FoundationQueryHead().to(device).eval()]
    else:
        heads, _ = load_radimagenet_bundle(heads_path, device)
        print(f"[+] Loaded {len(heads)} Fine-Tuned RadImageNet Query Head(s)")

    preds = []
    with torch.no_grad():
        for st in tqdm(studies, desc="RadImageNet Slicing & Inference"):
            # Mock or direct 4-slot slice extraction
            st_series = series_df[series_df["StudyInstanceUID"] == st] if not series_df.empty else pd.DataFrame()
            # Direct 4 slots x 8 slices extraction across [0.12, 0.88] depth band
            rad_vol, mask = extract_study_dicom_volumes(
                st, st_series, data_dir,
                slots=RADIMAGENET_SERIES_SLOTS,
                slice_budget=RADIMAGENET_DEFAULT_BUDGET,
                depth_band=RADIMAGENET_DEFAULT_BAND
            )
            rad_mask = mask.astype(np.float32)

            flat_slices = rad_vol.reshape(-1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            x_in = torch.from_numpy(flat_slices).to(device).float().div_(127.5).sub_(1.0)
            x_in = x_in.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feats = encoder(x_in)
            total_tokens = rad_vol.shape[0] * rad_vol.shape[1]
            token_feat = feats.view(1, total_tokens, -1)
            token_mask = torch.from_numpy(np.repeat(rad_mask, rad_vol.shape[1])).unsqueeze(0).to(device)

            head_probs = []
            for h in heads:
                logits = h(token_feat, token_mask)
                head_probs.append(torch.sigmoid(logits).cpu().numpy()[0])
            preds.append(np.mean(head_probs, axis=0))

    df_out = pd.DataFrame(preds, columns=TARGET_COLS)
    df_out.insert(0, "StudyInstanceUID", studies)
    return df_out


def main():
    args = parse_args()
    t_start = time.time()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"RSNA Knee Inference Engine | Active Device: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {data_dir}")

    # Resolve series metadata first to assist study discovery
    series_df = pd.DataFrame()
    if args.series_csv and Path(args.series_csv).exists():
        series_df = pd.read_csv(args.series_csv)
    elif (data_dir / "sample_series.csv").exists():
        series_df = pd.read_csv(data_dir / "sample_series.csv")
    elif (data_dir / "test_series.csv").exists():
        series_df = pd.read_csv(data_dir / "test_series.csv")
    elif (data_dir.parent / "test_series.csv").exists():
        series_df = pd.read_csv(data_dir.parent / "test_series.csv")

    studies = discover_studies(data_dir, series_df=series_df)
    if not studies:
        raise RuntimeError(f"No valid study directories found in {data_dir}!")
    print(f"Found {len(studies)} examination(s) to predict: {studies}")

    if series_df.empty:
        series_df = auto_build_series_metadata(data_dir, studies)

    xai_spec = args.xai
    xai_out_dir = Path(args.xai_out_dir) if args.xai else None

    if args.model == "convnext":
        df_sub = run_convnext_inference(
            studies, series_df, data_dir, Path(args.conv_ckpt), device,
            xai_spec=xai_spec, xai_out_dir=xai_out_dir
        )
    elif args.model == "radimagenet":
        df_sub = run_radimagenet_inference(
            studies, series_df, data_dir, Path(args.rad_ckpt), Path(args.resnet_ckpt), device
        )
        if xai_spec is not None:
            print("\n[*] Generating xAI explanations via ConvNeXt 6-View spatial attention backbone...")
            _ = run_convnext_inference(
                studies, series_df, data_dir, Path(args.conv_ckpt), device,
                xai_spec=xai_spec, xai_out_dir=xai_out_dir
            )
    else:
        df_conv = run_convnext_inference(
            studies, series_df, data_dir, Path(args.conv_ckpt), device,
            xai_spec=xai_spec, xai_out_dir=xai_out_dir
        )
        # Release GPU VRAM between models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        df_rad = run_radimagenet_inference(studies, series_df, data_dir, Path(args.rad_ckpt), Path(args.resnet_ckpt), device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        p_conv = df_conv[TARGET_COLS].values
        p_rad = df_rad[TARGET_COLS].values
        p_blended = blend_predictions(
            p_conv, p_rad,
            weight_conv=args.weight_conv,
            weight_rad=args.weight_rad,
            method=args.blend_method
        )
        df_sub = pd.DataFrame(p_blended, columns=TARGET_COLS)
        df_sub.insert(0, "StudyInstanceUID", studies)

    # Output verification & saving
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out_path, index=False)

    print("\n" + "="*75)
    print("  INFERENCE COMPLETED SUCCESSFULLY")
    print("="*75)
    print(f"Saved Submission to : {out_path.resolve()}")
    print(f"Total Studies       : {len(df_sub)}")
    print(f"Total Columns       : {len(df_sub.columns)}")
    print(f"Total Runtime       : {time.time() - t_start:.2f}s")
    print("="*75)
    print(df_sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
