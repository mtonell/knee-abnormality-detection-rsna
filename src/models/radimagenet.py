"""RadImageNet ResNet-50 Feature Extractor & Cross-Plane Attention Query Head."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet50

from src.constants import RADIMAGENET_SLOTS, NUM_CLASSES
from src.dicom import extract_study_dicom_volumes

logger = logging.getLogger(__name__)


class RadImageNetEncoder(nn.Module):
    """Official RadImageNet ResNet-50 medical transfer learning backbone.

    Pre-trained on 1.35M radiologist-annotated scans across 280 medical disease classes.
    Extracts 2048-dim global average pooled feature tokens per slice.
    """
    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            clean_state = {k.replace('backbone.', ''): v for k, v in state.items()}
            self.backbone.load_state_dict(clean_state, strict=True)
            logger.info("Loaded official RadImageNet ResNet-50 weights from %s", checkpoint_path)
        else:
            if checkpoint_path is not None:
                logger.warning("RadImageNet checkpoint %s not found. Initializing uninitialized ResNet-50.", checkpoint_path)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over 3-channel grayscale expanded slices."""
        return self.backbone(x).mean(dim=(2, 3))


class FoundationQueryHead(nn.Module):
    """Cross-Plane Multi-Head Query Attention Classifier for RadImageNet Tokens.

    Fuses 4 diverse series slots with 12 finding queries via Multi-Head Attention.
    """
    def __init__(
        self,
        n_slot: int = len(RADIMAGENET_SLOTS),
        cache_slices: int = 8,
        token_dim: int = 2048,
        head_dim: int = 512,
        n_targets: int = NUM_CLASSES
    ):
        super().__init__()
        self.n_slot = n_slot
        self.cache_slices = cache_slices
        self.head_dim = head_dim

        self.project = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, head_dim),
            nn.GELU()
        )
        self.plane = nn.Parameter(torch.randn(n_slot, head_dim) * 0.01)
        self.position = nn.Parameter(torch.randn(cache_slices, head_dim) * 0.01)
        self.query = nn.Parameter(torch.randn(n_targets, head_dim) * 0.02)
        self.attn = nn.MultiheadAttention(head_dim, 8, dropout=0.0, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(head_dim * 4),
            nn.Linear(head_dim * 4, head_dim),
            nn.GELU(),
            nn.Dropout(0.0)
        )
        self.weight = nn.Parameter(torch.randn(n_targets, head_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_targets))

    def forward(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cross-attention between 12 finding queries and acquired slice tokens."""
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.cache_slices, self.head_dim)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)

        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False

        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token, key_padding_mask=key_padding, need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, query.shape[1], -1)
        fused = self.fuse(torch.cat([attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


class RadImageNetCachedDataset(Dataset):
    """PyTorch Dataset delivering pre-extracted RadImageNet feature tokens directly from RAM.

    Eliminates all disk IO, DICOM parsing, and backbone compute during training epochs.
    """
    def __init__(
        self,
        features: np.ndarray,      # (N, 32, 2048) float32
        masks: np.ndarray,         # (N, 32) bool
        targets: np.ndarray,       # (N, 12) float32
        weights: np.ndarray,       # (N,) float32
        studies: List[str]
    ):
        self.features = features
        self.masks = masks
        self.targets = targets
        self.weights = weights
        self.studies = list(studies)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        feat = torch.from_numpy(self.features[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).bool()
        target = torch.from_numpy(self.targets[idx]).float()
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        study = self.studies[idx]
        return feat, mask, target, weight, study


def extract_or_load_radimagenet_cache(
    studies: List[str],
    data_dir: Path,
    series_df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    resnet_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    slots: List[str] = RADIMAGENET_SLOTS,
    slice_budget: Optional[Dict[str, int]] = None,
    depth_band: Tuple[float, float] = (0.12, 0.88),
    force_extract: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads cached 2048-dim RadImageNet features or extracts uncached studies on-the-fly.

    Returns:
        features: (N, 32, 2048) float32 numpy array
        masks: (N, 32) bool numpy array
        studies: list of N StudyInstanceUID strings
    """
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = slice_budget or {"SAG": 8, "COR": 8, "AX": 8}
    n_slices_per_slot = 8
    total_tokens = len(slots) * n_slices_per_slot  # 4 * 8 = 32

    cached_feats = {}
    cached_masks = {}

    if cache_path and Path(cache_path).exists() and not force_extract:
        try:
            with np.load(cache_path) as npz:
                f_arr = npz["features"]
                m_arr = npz["masks"]
                s_arr = [str(s) for s in npz["studies"]]
                for i, st in enumerate(s_arr):
                    cached_feats[st] = f_arr[i]
                    cached_masks[st] = m_arr[i]
            logger.info("Loaded %d study feature vectors from cache %s", len(s_arr), cache_path)
        except Exception as e:
            logger.warning("Could not read feature cache at %s (%s). Re-extracting.", cache_path, e)

    missing_studies = [st for st in studies if st not in cached_feats]

    if missing_studies:
        logger.info("Extracting RadImageNet features for %d study(ies)...", len(missing_studies))
        encoder = RadImageNetEncoder(resnet_path).to(dev).eval()

        for st in missing_studies:
            st_series = series_df[series_df["StudyInstanceUID"] == st] if not series_df.empty else pd.DataFrame()
            vols, series_mask = extract_study_dicom_volumes(
                st, st_series, data_dir,
                slots=slots,
                slice_budget=budget,
                depth_band=depth_band
            )  # vols: (4, 8, 224, 224) uint8, series_mask: (4,) bool

            # Expand 2D grayscale slices to 3-channel RGB for ResNet-50
            flat_slices = vols.reshape(-1, 224, 224)  # (32, 224, 224)
            x_in = torch.from_numpy(flat_slices).unsqueeze(1).repeat(1, 3, 1, 1).float().div_(255.0).to(dev)

            with torch.no_grad():
                slice_feats = encoder(x_in).cpu().numpy().astype(np.float32)  # (32, 2048)

            tok_mask = np.repeat(series_mask, n_slices_per_slot)  # (32,) bool
            cached_feats[st] = slice_feats
            cached_masks[st] = tok_mask

        # Save or update cache
        if cache_path:
            out_p = Path(cache_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            all_uids = list(cached_feats.keys())
            all_feats = np.stack([cached_feats[u] for u in all_uids], axis=0)
            all_masks = np.stack([cached_masks[u] for u in all_uids], axis=0)
            np.savez_compressed(
                out_p,
                features=all_feats,
                masks=all_masks,
                studies=np.array(all_uids, dtype=object)
            )
            logger.info("Saved consolidated RadImageNet feature cache (%d studies) to %s", len(all_uids), out_p)

    out_feats = np.stack([cached_feats[st] for st in studies], axis=0).astype(np.float32)
    out_masks = np.stack([cached_masks[st] for st in studies], axis=0).astype(bool)
    return out_feats, out_masks, studies


def load_radimagenet_bundle(
    heads_path: Path,
    device: torch.device
) -> Tuple[List[FoundationQueryHead], List[dict]]:
    """Unpacks query head checkpoint bundle (supports multi-fold ensemble or single head)."""
    bundle = torch.load(heads_path, map_location="cpu", weights_only=False)
    all_folds = bundle.get('folds', [])

    if not all_folds and 'state_dict' in bundle:
        all_folds = [{'fold': 0, 'state_dict': bundle['state_dict']}]
    elif not all_folds and isinstance(bundle, dict):
        all_folds = [{'fold': 0, 'state_dict': bundle}]

    heads = []
    for f_rec in all_folds:
        h = FoundationQueryHead().to(device)
        st_dict = f_rec.get("state_dict", f_rec)
        h.load_state_dict(st_dict, strict=True)
        h.eval()
        heads.append(h)

    return heads, all_folds
