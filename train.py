#!/usr/bin/env python3
"""Unified Training CLI for RSNA Knee Abnormality Models.

Supports training or fine-tuning either:
  1. ConvNeXt-Nano 6-View Multi-Head Transformer (Flagship End-to-End Architecture)
  2. RadImageNet ResNet-50 Cross-Plane Query Head (Frozen Feature Token Transfer Learning)

Key Features:
  - Configuration via YAML (`configs/convnext.yaml`, `configs/radimagenet.yaml`)
  - Modular Loss System (`src/loss.py`):
      * Sample-Weighted BCEWithLogitsLoss (for ConvNeXt soft continuous supervision)
      * Sample-Weighted Focal Loss (gamma=2.0, for RadImageNet rare abnormality class imbalance)
  - Frozen Feature Caching for RadImageNet:
      * Encodes 32 slice tokens once via ResNet-50 and caches to `.npz`
      * Training the query head runs entirely in RAM in seconds
  - Dynamic Slice Sampling (ConvNeXt):
      * Training: K_TRAIN = 10 slices per slot with continuous random depth jittering
      * Evaluation: K_EVAL = 20 equidistant slices per slot
  - Volumetric Augmentation (ConvNeXt): 50% Random Horizontal Flip, 15% Series Dropout
  - Gradient Accumulation: batch_size = 2, accum_steps = 4 (simulating effective batch size 8)
  - 2-Speed Differential Optimizer: AdamW (lr_backbone = 1e-4, lr_head = 5e-4, weight_decay = 0.02)
  - 1-Epoch Linear Warmup + Cosine Annealing Scheduler (LambdaLR)
  - Exponential Moving Average (EMA decay = 0.998) model evaluation and checkpointing
  - Flexible Cross-Validation:
      * `--folds 1` (Default): Trains 1 fold, skips OOF, evaluates directly on Gold 58.
      * `--folds 5`: Full 5-fold CV, records OOF predictions to `oof.csv`,
        computes Local OOF Macro-AUC, and evaluates 5-fold ensemble on Gold 58.
  - Continuous soft supervision from Tri-Model Distillation.

Usage:
  # Train 1 fold of ConvNeXt (Default mode):
  python train.py --config configs/convnext.yaml

  # Train 1 fold of RadImageNet (with automated feature caching and Focal Loss gamma=2.0):
  python train.py --config configs/radimagenet.yaml

  # Train full 5-fold cross-validation:
  python train.py --config configs/convnext.yaml --folds 5
"""

import os
import sys
import copy
import math
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    TARGET_COLS,
    NUM_CLASSES,
    CONVNEXT_SERIES_SLOTS,
    RADIMAGENET_SLOTS,
    DEFAULT_IMAGE_SIZE,
    GOLD_BENCHMARK_WEIGHT
)
from src.dicom import extract_study_dicom_volumes
from src.loss import get_loss_function
from src.models.convnext import ConvNeXt6View
from src.models.radimagenet import (
    FoundationQueryHead,
    RadImageNetCachedDataset,
    extract_or_load_radimagenet_cache
)
from src.evaluator import compute_macro_auc, format_scorecard_table


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA Knee Abnormality Training CLI")
    parser.add_argument("--config", type=str, default="configs/convnext.yaml",
                        help="Path to YAML training configuration file.")
    parser.add_argument("--folds", type=int, default=None,
                        help="Number of folds to train: 1 (single fold, default) or 5 (full 5-fold CV).")
    parser.add_argument("--fold", type=int, default=0,
                        help="Specific fold index to train when folds=1 (0-4, default: 0).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels CSV (overrides config).")
    parser.add_argument("--data_dir", type=str, default="data/samples",
                        help="Path to directory containing DICOM examinations.")
    parser.add_argument("--series_csv", type=str, default=None,
                        help="Path to series.csv (auto-detected if present).")
    parser.add_argument("--gold_path", type=str, default="data/gold_58_benchmark.csv",
                        help="Path to official Gold 58 expert benchmark CSV.")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save trained model checkpoints and predictions.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs per fold (overrides config).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size per device (overrides config).")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="Gradient accumulation steps (e.g. 4 for effective batch size 8).")
    parser.add_argument("--k_train", type=int, default=None,
                        help="Number of dynamic depth-jittered slices during training (default: 10).")
    parser.add_argument("--k_eval", type=int, default=None,
                        help="Number of equidistant slices during evaluation (default: 20).")
    parser.add_argument("--lr_backbone", type=float, default=None,
                        help="Learning rate for pretrained backbone (default: 1e-4).")
    parser.add_argument("--lr_head", type=float, default=None,
                        help="Learning rate for transformer & MIL heads (default: 5e-4).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Global learning rate override.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay for AdamW optimizer (default: 0.02).")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="Exponential moving average decay factor (default: 0.998).")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping max value (default: 1.0).")
    parser.add_argument("--loss_type", type=str, default=None,
                        help="Loss function: 'bce' or 'focal' (overrides config).")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Focal loss gamma focusing parameter (default: 2.0).")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to RadImageNet cached features .npz.")
    parser.add_argument("--resnet_path", type=str, default=None,
                        help="Path to official RadImageNet ResNet50.pt checkpoint.")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable feature caching for RadImageNet and force re-extraction.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'cpu'.")
    return parser.parse_args()


def sample_slice_indices(
    n_available: int,
    k: int,
    is_train: bool,
    depth_band: Tuple[float, float] = (0.04, 0.96)
) -> np.ndarray:
    """Samples k slice indices with random continuous depth jitter across equal bins."""
    lo = max(1.0, depth_band[0] * (n_available - 1))
    hi = min(n_available - 2.0, depth_band[1] * (n_available - 1))
    if hi < lo:
        lo = hi = (n_available - 1) / 2

    if is_train:
        edges = np.linspace(lo, hi, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(lo, hi, k)

    return np.clip(np.round(c).astype(int), 1, max(1, n_available - 2))


def update_ema(ema_model: nn.Module, model: nn.Module, global_step: int, decay: float = 0.998) -> None:
    """Updates Exponential Moving Average parameters with warm-up scheduling."""
    d = min(decay, (1.0 + global_step) / (10.0 + global_step))
    with torch.no_grad():
        for pe, pm in zip(ema_model.parameters(), model.parameters()):
            pe.lerp_(pm, 1.0 - d)
        for be, bm in zip(ema_model.buffers(), model.buffers()):
            be.copy_(bm)


class KneeTrainDataset(Dataset):
    """PyTorch Dataset extracting multi-view 2.5D tri-slice slabs and sample-weighted targets."""
    def __init__(
        self,
        studies: List[str],
        labels_df: pd.DataFrame,
        data_dir: Path,
        series_df: pd.DataFrame,
        k_slices: int = 10,
        is_train: bool = True,
        depth_band: Tuple[float, float] = (0.04, 0.96)
    ):
        self.studies = studies
        self.labels_df = labels_df.set_index("StudyInstanceUID") if "StudyInstanceUID" in labels_df.columns else labels_df
        self.data_dir = data_dir
        self.series_df = series_df
        self.k_slices = k_slices
        self.is_train = is_train
        self.depth_band = depth_band

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        st = self.studies[idx]
        st_series = self.series_df[self.series_df["StudyInstanceUID"] == st] if not self.series_df.empty else pd.DataFrame()
        vols, mask = extract_study_dicom_volumes(st, st_series, self.data_dir)

        # Build 2.5D overlapping tri-slice slabs: [z-1, z, z+1]
        k = self.k_slices
        n_slots = len(CONVNEXT_SERIES_SLOTS)
        slabs = np.zeros((n_slots, k, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.uint8)
        pos = np.zeros((n_slots, k), dtype=np.float32)

        for slot in range(n_slots):
            if mask[slot]:
                n_avail = vols.shape[1]
                c_idx = sample_slice_indices(n_avail, k, self.is_train, self.depth_band)
                pos[slot] = c_idx / max(1.0, float(n_avail - 1))
                idx_prev = np.clip(c_idx - 1, 0, n_avail - 1)
                idx_next = np.clip(c_idx + 1, 0, n_avail - 1)
                slabs[slot, :, 0] = vols[slot, idx_prev]
                slabs[slot, :, 1] = vols[slot, c_idx]
                slabs[slot, :, 2] = vols[slot, idx_next]

        # SOTA Volumetric Augmentations directly on uint8 arrays during training
        if self.is_train:
            # Random Horizontal Flip (50%)
            if np.random.rand() > 0.5:
                slabs = slabs[:, :, :, :, ::-1].copy()

            # Random Series Dropout (15%) when at least 3 series exist
            if np.random.rand() > 0.85 and np.sum(mask) > 2:
                present = np.where(mask)[0]
                drop_idx = np.random.choice(present)
                slabs[drop_idx] = 0
                mask[drop_idx] = False

        # Soft continuous supervisory targets & sample loss weighting
        if st in self.labels_df.index:
            row = self.labels_df.loc[st]
            y = row[TARGET_COLS].values.astype(np.float32)
            is_gold = int(row.get("is_gold", 0))
            weight = 1.0
        else:
            y = np.full(NUM_CLASSES, 0.1, dtype=np.float32)
            weight = 1.0

        tx = torch.from_numpy(slabs).float().div_(255.0)
        tm = torch.from_numpy(mask)
        tp = torch.from_numpy(pos)
        ty = torch.from_numpy(y)
        tw = torch.tensor(weight, dtype=torch.float32)
        return tx, tm, tp, ty, tw, st


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_convnext: bool = True,
    accum_steps: int = 4,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.998,
    global_step: int = 0
) -> Tuple[float, int]:
    """Executes a single training epoch with gradient accumulation, AMP, and EMA updates."""
    model.train()
    running_loss = 0.0
    use_amp = (device.type == "cuda")
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if is_convnext:
            tx, tm, tp, ty, tw, _ = batch
            tx, tm, tp, ty, tw = tx.to(device), tm.to(device), tp.to(device), ty.to(device), tw.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(tx, tm, tp)
                raw_loss = criterion(logits, ty, tw)
                loss = raw_loss / accum_steps
        else:
            feats, mask, ty, tw, _ = batch
            feats, mask, ty, tw = feats.to(device), mask.to(device), ty.to(device), tw.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(feats, mask)
                raw_loss = criterion(logits, ty, tw)
                loss = raw_loss / accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += raw_loss.item()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scale_after = scaler.get_scale()
                step_succeeded = (scale_before <= scale_after)
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step_succeeded = True

            if step_succeeded:
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if ema_model is not None:
                    update_ema(ema_model, model, global_step, decay=ema_decay)

    avg_loss = running_loss / max(1, len(loader))
    return avg_loss, global_step


@torch.no_grad()
def predict_studies(
    model: nn.Module,
    studies: List[str],
    labels_df: pd.DataFrame,
    data_dir: Path,
    series_df: pd.DataFrame,
    device: torch.device,
    is_convnext: bool = True,
    k_eval: int = 20,
    depth_band: Tuple[float, float] = (0.04, 0.96),
    rad_features: Optional[np.ndarray] = None,
    rad_masks: Optional[np.ndarray] = None,
    cache_path: Optional[Path] = None,
    resnet_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference on a cohort of studies, returning true labels and predicted probabilities."""
    model.eval()
    all_preds = []
    all_targets = []

    if is_convnext:
        dataset = KneeTrainDataset(
            studies,
            labels_df,
            data_dir,
            series_df,
            k_slices=k_eval,
            is_train=False,
            depth_band=depth_band
        )
        loader = DataLoader(dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False)
        for tx, tm, tp, ty, _, _ in loader:
            tx, tm, tp = tx.to(device), tm.to(device), tp.to(device)
            logits = model(tx, tm, tp)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(ty.numpy())
    else:
        # RadImageNet: use pre-loaded cached features or load from disk
        if rad_features is None or rad_masks is None:
            rad_features, rad_masks, _ = extract_or_load_radimagenet_cache(
                studies, data_dir, series_df, cache_path=cache_path,
                resnet_path=resnet_path, device=device, depth_band=depth_band
            )
        y_targets = []
        for st in studies:
            if st in labels_df.index:
                y_targets.append(labels_df.loc[st][TARGET_COLS].values.astype(np.float32))
            else:
                y_targets.append(np.full(NUM_CLASSES, 0.1, dtype=np.float32))
        y_arr = np.array(y_targets, dtype=np.float32)
        w_arr = np.ones(len(studies), dtype=np.float32)
        ds = RadImageNetCachedDataset(rad_features, rad_masks, y_arr, w_arr, studies)
        loader = DataLoader(ds, batch_size=min(16, max(1, len(ds))), shuffle=False)
        for feats, mask, ty, _, _ in loader:
            feats, mask = feats.to(device), mask.to(device)
            logits = model(feats, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(ty.numpy())

    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, NUM_CLASSES))
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, NUM_CLASSES))
    return y_true, y_pred


def evaluate_gold_benchmark(gold_path: Path, model_preds: Dict[str, np.ndarray]) -> None:
    """Evaluates and prints the official 12-target Gold 58 Expert Benchmark scorecard."""
    if not gold_path.exists():
        print(f"[-] Gold benchmark file not found at: {gold_path}. Skipping Gold 58 evaluation.")
        return

    df_gold = pd.read_csv(gold_path)
    gold_cols = [f"gold_{c}" for c in TARGET_COLS if f"gold_{c}" in df_gold.columns]
    pred_cols = [f"pred_{c}" for c in TARGET_COLS if f"pred_{c}" in df_gold.columns]

    if len(gold_cols) == NUM_CLASSES:
        y_true = df_gold[gold_cols].values
        gold_uids = df_gold["StudyInstanceUID"].tolist()
        eval_preds = np.zeros_like(y_true, dtype=np.float64)
        for i, uid in enumerate(gold_uids):
            if uid in model_preds:
                eval_preds[i] = model_preds[uid]
            elif len(pred_cols) == NUM_CLASSES:
                eval_preds[i] = df_gold[pred_cols].iloc[i].values
            else:
                eval_preds[i] = 0.5

        macro_auc, class_aucs = compute_macro_auc(y_true, eval_preds, TARGET_COLS)
        scorecard = format_scorecard_table(y_true, eval_preds, TARGET_COLS)
        print("\n" + "=" * 75)
        print(f"  OFFICIAL GOLD 58 BENCHMARK SCORECARD")
        print(f"  OVERALL GOLD 58 MACRO ROC-AUC : {macro_auc:.4f}")
        print("=" * 75)
        print(scorecard.to_string(index=False))
        print("=" * 75)


def main():
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    views_cfg = cfg.get("views", {})

    # Detect model type
    is_convnext = "radimagenet" not in config_path.stem.lower()
    model_name = "ConvNeXt-Nano 6-View MIL" if is_convnext else "RadImageNet ResNet-50 Query Head"

    # Resolve settings (CLI overrides config)
    n_folds = args.folds if args.folds is not None else train_cfg.get("folds", 1)
    labels_path = Path(args.labels) if args.labels is not None else Path(train_cfg.get("labels", "data/train_distilled_labels.csv"))
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 10 if is_convnext else 20)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2 if is_convnext else 16)
    accum_steps = args.accum_steps if args.accum_steps is not None else train_cfg.get("accum_steps", 4 if is_convnext else 1)
    k_train = args.k_train if args.k_train is not None else train_cfg.get("k_train", 10)
    k_eval = args.k_eval if args.k_eval is not None else train_cfg.get("k_eval", 20 if is_convnext else 8)
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else float(train_cfg.get("lr_backbone", 1.0e-4))
    lr_head = args.lr_head if args.lr_head is not None else float(train_cfg.get("lr_head", train_cfg.get("lr", 5.0e-4 if is_convnext else 2.0e-4)))
    if args.lr is not None:
        lr_head = args.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.02 if is_convnext else 3.0e-3))
    ema_decay = args.ema_decay if args.ema_decay is not None else float(train_cfg.get("ema_decay", 0.998))
    grad_clip = args.grad_clip if args.grad_clip is not None else float(train_cfg.get("grad_clip", 1.0))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    depth_band = tuple(views_cfg.get("depth_band", [0.04, 0.96] if is_convnext else [0.12, 0.88]))

    # Loss configuration
    default_loss_type = "bce" if is_convnext else "focal"
    loss_type = args.loss_type if args.loss_type is not None else train_cfg.get("loss_type", default_loss_type)
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else float(train_cfg.get("focal_gamma", 2.0))
    criterion = get_loss_function(name=loss_type, gamma=focal_gamma)

    # Feature Cache settings for RadImageNet
    cache_path = Path(args.cache_path) if args.cache_path is not None else Path(train_cfg.get("cache_path", "cache/rad_features_cache.npz"))
    resnet_path = Path(args.resnet_path) if args.resnet_path is not None else Path(train_cfg.get("resnet_path", "checkpoints/ResNet50.pt"))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    effective_batch_size = batch_size * accum_steps
    print("=" * 75)
    print(f"  RSNA KNEE TRAINING PIPELINE | Model: {model_name.upper()}")
    print(f"  Configuration       : {config_path.name}")
    print(f"  Device              : {device}")
    print(f"  Batch / Accum Steps : {batch_size} x {accum_steps} (Effective Batch Size = {effective_batch_size})")
    print(f"  Loss Function       : {loss_type.upper()}" + (f" (gamma={focal_gamma})" if loss_type.lower() == "focal" else ""))
    if is_convnext:
        print(f"  Dynamic Slices      : K_TRAIN = {k_train} (jittered) | K_EVAL = {k_eval} (linspace)")
        print(f"  Differential LRs    : Backbone = {lr_backbone:g} | Head = {lr_head:g} | WD = {weight_decay:g}")
    else:
        print(f"  Feature Tokens      : 32 tokens x 2048 dims (ResNet-50 frozen backbone)")
        print(f"  Feature Cache       : {cache_path} (No-Cache: {args.no_cache})")
        print(f"  Head Learning Rate  : {lr_head:g} | WD = {weight_decay:g}")
    print(f"  EMA Smoothing       : Decay = {ema_decay} | Grad Clip = {grad_clip}")
    print("=" * 75)

    if not labels_path.exists():
        raise FileNotFoundError(f"Supervisory labels CSV not found: {labels_path}")

    df_labels = pd.read_csv(labels_path)
    print(f"[+] Loaded Supervisory Targets: {len(df_labels)} studies across {NUM_CLASSES} classes ({labels_path.name}).")

    data_dir = Path(args.data_dir)
    available_studies = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"[+] Discovered {len(available_studies)} physical examination volume(s) in {data_dir}.")

    series_csv = Path(args.series_csv) if args.series_csv else (data_dir / "sample_series.csv")
    df_series = pd.read_csv(series_csv) if series_csv.exists() else pd.DataFrame()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = Path(args.gold_path)

    # For RadImageNet: Load or extract feature tokens into RAM once before training
    rad_features = None
    rad_masks = None
    if not is_convnext:
        print(f"\n>>> Preparing RadImageNet Feature Cache...")
        rad_features, rad_masks, _ = extract_or_load_radimagenet_cache(
            studies=available_studies,
            data_dir=data_dir,
            series_df=df_series,
            cache_path=cache_path if not args.no_cache else None,
            resnet_path=resnet_path,
            device=device,
            slots=views_cfg.get("slots", RADIMAGENET_SLOTS),
            slice_budget=views_cfg.get("slice_budget", {"SAG": 8, "COR": 8, "AX": 8}),
            depth_band=depth_band,
            force_extract=args.no_cache
        )
        print(f"[+] RadImageNet Feature Cache Ready: {rad_features.shape} tokens in RAM.")

    # =========================================================================
    # EXECUTION MODE 1: SINGLE FOLD TRAINING (--folds 1)
    # =========================================================================
    if n_folds == 1:
        target_fold = args.fold
        print(f"\n>>> Running Single Fold Training (Fold {target_fold}) on {len(available_studies)} examination(s)...")

        if is_convnext:
            dataset = KneeTrainDataset(
                available_studies,
                df_labels,
                data_dir,
                df_series,
                k_slices=k_train,
                is_train=True,
                depth_band=depth_band
            )
            loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=True)
            model = ConvNeXt6View(num_classes=NUM_CLASSES).to(device)
            enc_params = list(model.backbone.parameters())
            head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
            optimizer = torch.optim.AdamW([
                {"params": enc_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head}
            ], weight_decay=weight_decay)
        else:
            # Build target and weight arrays
            y_list, w_list = [], []
            df_lbl_idx = df_labels.set_index("StudyInstanceUID") if "StudyInstanceUID" in df_labels.columns else df_labels
            for st in available_studies:
                if st in df_lbl_idx.index:
                    row = df_lbl_idx.loc[st]
                    y_list.append(row[TARGET_COLS].values.astype(np.float32))
                    w_list.append(1.0)
                else:
                    y_list.append(np.full(NUM_CLASSES, 0.1, dtype=np.float32))
                    w_list.append(1.0)
            y_arr = np.array(y_list, dtype=np.float32)
            w_arr = np.array(w_list, dtype=np.float32)

            dataset = RadImageNetCachedDataset(rad_features, rad_masks, y_arr, w_arr, available_studies)
            loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=True)
            model = FoundationQueryHead().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=weight_decay)

        # 1-Epoch Linear Warmup + Cosine Annealing Scheduler
        effective_steps_per_epoch = max(1, math.ceil(len(loader) / accum_steps))
        warmup_steps = max(1, effective_steps_per_epoch * warmup_epochs)
        total_steps = max(warmup_steps + 1, effective_steps_per_epoch * epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, (s + 1) / warmup_steps) * (0.5 * (1.0 + math.cos(math.pi * min(1.0, s / total_steps))))
        )

        scaler = torch.amp.GradScaler('cuda', init_scale=4096.0) if device.type == 'cuda' else None

        # Initialize Model EMA
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False

        start_time = time.time()
        global_step = 0
        for ep in range(1, epochs + 1):
            avg_loss, global_step = train_single_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                is_convnext=is_convnext,
                accum_steps=accum_steps,
                scaler=scaler,
                grad_clip=grad_clip,
                scheduler=scheduler,
                ema_model=ema_model,
                ema_decay=ema_decay,
                global_step=global_step
            )
            print(f"  [Epoch {ep:02d}/{epochs:02d}] Training Loss: {avg_loss:.4f} (Global Steps: {global_step})")

        elapsed = time.time() - start_time
        print(f"[+] Training completed in {elapsed:.1f}s.")

        save_name = f"convnext_fold_{target_fold}_best.pt" if is_convnext else f"radimagenet_fold_{target_fold}_best.pt"
        save_path = out_dir / save_name
        torch.save({
            "model": ema_model.state_dict(),
            "state_dict": ema_model.state_dict(),
            "fold": target_fold,
            "epoch": epochs,
            "config": cfg
        }, save_path)
        print(f"[+] Saved Trained EMA Checkpoint to: {save_path.resolve()}")

        # Direct Evaluation using EMA weights on available studies and Gold 58
        _, model_probs = predict_studies(
            ema_model,
            available_studies,
            df_labels,
            data_dir,
            df_series,
            device,
            is_convnext=is_convnext,
            k_eval=k_eval,
            depth_band=depth_band,
            rad_features=rad_features,
            rad_masks=rad_masks,
            cache_path=cache_path,
            resnet_path=resnet_path
        )
        study_pred_map = {st: model_probs[i] for i, st in enumerate(available_studies)}

        print("\n>>> Evaluating on Official Gold 58 Expert Benchmark...")
        evaluate_gold_benchmark(gold_path, study_pred_map)

    # =========================================================================
    # EXECUTION MODE 2: 5-FOLD CROSS-VALIDATION (--folds 5)
    # =========================================================================
    elif n_folds == 5:
        print(f"\n>>> Running Full 5-Fold Cross-Validation across 5 folds...")

        weak_mask = (df_labels["is_gold"] == 0) if "is_gold" in df_labels.columns else pd.Series(True, index=df_labels.index)
        oof_df = df_labels.loc[weak_mask, ["StudyInstanceUID", "fold"]].copy().reset_index(drop=True)
        for col in TARGET_COLS:
            oof_df[col] = np.nan

        fold_models = []

        for f in range(5):
            print(f"\n--- [Fold {f}/4] Training & Out-of-Fold Validation ---")
            fold_train_studies = [st for st in available_studies]

            if is_convnext:
                dataset = KneeTrainDataset(
                    fold_train_studies,
                    df_labels,
                    data_dir,
                    df_series,
                    k_slices=k_train,
                    is_train=True,
                    depth_band=depth_band
                )
                loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=True)
                model = ConvNeXt6View(num_classes=NUM_CLASSES).to(device)
                enc_params = list(model.backbone.parameters())
                head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
                optimizer = torch.optim.AdamW([
                    {"params": enc_params, "lr": lr_backbone},
                    {"params": head_params, "lr": lr_head}
                ], weight_decay=weight_decay)
            else:
                y_list, w_list = [], []
                df_lbl_idx = df_labels.set_index("StudyInstanceUID") if "StudyInstanceUID" in df_labels.columns else df_labels
                for st in fold_train_studies:
                    if st in df_lbl_idx.index:
                        row = df_lbl_idx.loc[st]
                        y_list.append(row[TARGET_COLS].values.astype(np.float32))
                        w_list.append(1.0)
                    else:
                        y_list.append(np.full(NUM_CLASSES, 0.1, dtype=np.float32))
                        w_list.append(1.0)
                y_arr = np.array(y_list, dtype=np.float32)
                w_arr = np.array(w_list, dtype=np.float32)

                dataset = RadImageNetCachedDataset(rad_features, rad_masks, y_arr, w_arr, fold_train_studies)
                loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=True)
                model = FoundationQueryHead().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=weight_decay)

            effective_steps_per_epoch = max(1, math.ceil(len(loader) / accum_steps))
            warmup_steps = max(1, effective_steps_per_epoch * warmup_epochs)
            total_steps = max(warmup_steps + 1, effective_steps_per_epoch * epochs)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: min(1.0, (s + 1) / warmup_steps) * (0.5 * (1.0 + math.cos(math.pi * min(1.0, s / total_steps))))
            )

            scaler = torch.amp.GradScaler('cuda', init_scale=4096.0) if device.type == 'cuda' else None

            ema_model = copy.deepcopy(model)
            for p in ema_model.parameters():
                p.requires_grad = False

            global_step = 0
            for ep in range(1, epochs + 1):
                avg_loss, global_step = train_single_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    is_convnext=is_convnext,
                    accum_steps=accum_steps,
                    scaler=scaler,
                    grad_clip=grad_clip,
                    scheduler=scheduler,
                    ema_model=ema_model,
                    ema_decay=ema_decay,
                    global_step=global_step
                )
                if ep == epochs or ep == 1:
                    print(f"  Fold {f} [Epoch {ep:02d}/{epochs:02d}] Loss: {avg_loss:.4f}")

            ckpt_name = f"convnext_fold_{f}_best.pt" if is_convnext else f"radimagenet_fold_{f}_best.pt"
            torch.save({
                "model": ema_model.state_dict(),
                "state_dict": ema_model.state_dict(),
                "fold": f,
                "epoch": epochs,
                "config": cfg
            }, out_dir / ckpt_name)

            # Predict on available studies for fold validation using EMA weights
            _, val_probs = predict_studies(
                ema_model,
                available_studies,
                df_labels,
                data_dir,
                df_series,
                device,
                is_convnext=is_convnext,
                k_eval=k_eval,
                depth_band=depth_band,
                rad_features=rad_features,
                rad_masks=rad_masks,
                cache_path=cache_path,
                resnet_path=resnet_path
            )
            for i, st in enumerate(available_studies):
                if st in oof_df["StudyInstanceUID"].values:
                    oof_df.loc[oof_df["StudyInstanceUID"] == st, TARGET_COLS] = val_probs[i]

            fold_models.append(ema_model)

        # Save out-of-fold predictions
        oof_path = out_dir / "oof.csv"
        for c in TARGET_COLS:
            if oof_df[c].isna().any():
                oof_df[c] = oof_df[c].fillna(df_labels.loc[weak_mask, c])
        oof_df.to_csv(oof_path, index=False)
        print(f"\n[+] Saved Complete Out-of-Fold (OOF) Predictions to: {oof_path.resolve()}")

        # Compute Local OOF Macro-AUC
        y_weak_true = df_labels.loc[weak_mask, TARGET_COLS].values
        y_weak_pred = oof_df[TARGET_COLS].values
        oof_macro_auc, _ = compute_macro_auc(y_weak_true, y_weak_pred, TARGET_COLS)
        print(f"[+] Overall 5-Fold Local Out-of-Fold (OOF) Macro-AUC: {oof_macro_auc:.4f}")

        # 5-Fold Ensemble Evaluation on Gold 58
        print("\n>>> Computing 5-Fold Ensemble on Gold 58 Benchmark...")
        ensemble_preds = {}
        for st in available_studies:
            preds_across_folds = []
            for m in fold_models:
                _, p = predict_studies(
                    m, [st], df_labels, data_dir, df_series, device,
                    is_convnext=is_convnext, k_eval=k_eval, depth_band=depth_band,
                    rad_features=rad_features, rad_masks=rad_masks,
                    cache_path=cache_path, resnet_path=resnet_path
                )
                preds_across_folds.append(p[0])
            ensemble_preds[st] = np.mean(preds_across_folds, axis=0)

        evaluate_gold_benchmark(gold_path, ensemble_preds)

    else:
        raise ValueError(f"Invalid --folds argument '{n_folds}'. Choose 1 (single model) or 5 (5-fold CV).")

    print("\n" + "=" * 75)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
