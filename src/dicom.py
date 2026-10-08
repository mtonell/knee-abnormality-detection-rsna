"""DICOM reading, intensity windowing, and multi-sequence series extraction."""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pydicom

from src.constants import (
    PLANE_MAP,
    CONVNEXT_SERIES_SLOTS,
    DEFAULT_IMAGE_SIZE
)
from src.geometry import (
    parse_series_geometry,
    crop_and_pad_square,
    apply_laterality_normalization
)


def read_dicom_pixels(dcm_path: Path) -> np.ndarray:
    """Reads raw DICOM pixel array, applies rescale slope/intercept, and handles MONOCHROME1."""
    d = pydicom.dcmread(dcm_path)
    px = d.pixel_array.astype(np.float32)
    if px.ndim == 3:
        px = px[px.shape[0] // 2]
    slope = float(getattr(d, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(d, "RescaleIntercept", 0.0) or 0.0)
    px = px * slope + intercept
    if str(getattr(d, "PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1":
        px = px.max() - px
    return px


def percentile_window_uint8(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Dynamic foreground percentile intensity windowing to uint8 [0, 255].

    Suppresses extreme fat signals and RF surface coil gain artifacts.
    """
    pos = arr[arr > 0]
    if len(pos) < 64:
        v_min, v_max = float(arr.min()), float(arr.max())
    else:
        v_min = float(np.percentile(pos, p_low))
        v_max = float(np.percentile(pos, p_high))
    if v_max <= v_min:
        v_max = v_min + 1.0
    clipped = np.clip(arr, v_min, v_max)
    return ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)


def get_series_slot_name(plane_str: str, fat_suppression: int) -> Optional[str]:
    """Maps anatomical plane string and fat suppression flag to canonical 6-slot code."""
    p_code = PLANE_MAP.get(str(plane_str).strip().lower(), None)
    if not p_code:
        return None
    fs_code = "FS" if int(fat_suppression) == 1 else "N"
    return f"{p_code}_{fs_code}"


DEFAULT_SLICE_BUDGET: Dict[str, int] = {"SAG": 20, "COR": 20, "AX": 16}
DEFAULT_DEPTH_BAND: Tuple[float, float] = (0.04, 0.96)


def process_series_to_volume(
    files: List[Path],
    spacing: Tuple[float, float],
    plane_code: str,
    target_depth: int = 20,
    depth_band: Tuple[float, float] = (0.04, 0.96),
    is_right_knee: bool = False,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Interpolates slices across depth_band to target depth, applies square millimeter crop & laterality normalization."""
    n_files = len(files)
    if n_files == 0:
        return np.zeros((target_depth, image_size, image_size), dtype=np.uint8)

    lo = int(round(depth_band[0] * (n_files - 1)))
    hi = int(round(depth_band[1] * (n_files - 1)))
    if hi < lo:
        lo = hi = (n_files - 1) // 2

    indices = np.round(np.linspace(lo, hi, target_depth)).astype(int)
    volume = np.zeros((target_depth, image_size, image_size), dtype=np.uint8)
    for i, idx in enumerate(indices):
        try:
            raw_px = read_dicom_pixels(files[idx])
            norm_px = percentile_window_uint8(raw_px)
            volume[i] = crop_and_pad_square(norm_px, spacing, plane_code, out_px=image_size)
        except Exception:
            pass

    return apply_laterality_normalization(volume, plane_code, is_right_knee)


def extract_study_dicom_volumes(
    study_id: str,
    study_series_df: pd.DataFrame,
    series_dir: Path,
    slots: Optional[List[str]] = None,
    slice_budget: Optional[Dict[str, int]] = None,
    depth_band: Optional[Tuple[float, float]] = None,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds (len(slots), max_k, H, W) uint8 multi-view volume and boolean presence mask directly from DICOMs."""
    slots = slots if slots is not None else CONVNEXT_SERIES_SLOTS
    slice_budget = slice_budget if slice_budget is not None else DEFAULT_SLICE_BUDGET
    depth_band = depth_band if depth_band is not None else DEFAULT_DEPTH_BAND

    best_series_by_slot: Dict[str, dict] = {}
    
    # If study_series_df is empty, attempt on-the-fly series discovery from directory structure
    if study_series_df.empty:
        study_subdirs = []
        possible_roots = [
            series_dir / str(study_id),
            series_dir / "test_series" / str(study_id),
            series_dir / "train_series" / str(study_id),
            series_dir if series_dir.name == str(study_id) else None
        ]
        for pr in possible_roots:
            if pr is not None and pr.exists() and pr.is_dir():
                study_subdirs.extend([p for p in pr.iterdir() if p.is_dir()])
        
        fallback_rows = []
        for s_folder in study_subdirs:
            dcms = list(s_folder.glob("*.dcm"))
            if dcms:
                try:
                    h = pydicom.dcmread(dcms[0], stop_before_pixels=True)
                    desc = str(getattr(h, "SeriesDescription", "") or "").lower()
                    plane = "Sagittal" if "sag" in desc else ("Coronal" if "cor" in desc else ("Axial" if "ax" in desc else "Sagittal"))
                    fs = 1 if any(t in desc for t in ["fs", "fat", "stir", "spair", "water"]) else 0
                    fallback_rows.append({
                        "SeriesInstanceUID": s_folder.name,
                        "Anatomical_Plane": plane,
                        "Fat_Suppression": fs
                    })
                except Exception:
                    pass
        if fallback_rows:
            study_series_df = pd.DataFrame(fallback_rows)

    for _, s_row in study_series_df.iterrows():
        plane = str(s_row.get("Anatomical_Plane", ""))
        fs = int(s_row.get("Fat_Suppression", 0))
        slot = get_series_slot_name(plane, fs)
        if not slot:
            continue
        series_uid = str(s_row["SeriesInstanceUID"])
        candidates = [
            series_dir / str(study_id) / series_uid,
            series_dir / series_uid,
            series_dir / "test_series" / str(study_id) / series_uid,
            series_dir / "test_series" / series_uid,
            series_dir / "train_series" / str(study_id) / series_uid,
            series_dir / "train_series" / series_uid,
            series_dir.parent / "test_series" / str(study_id) / series_uid,
            series_dir.parent / "test_series" / series_uid,
        ]
        s_dir = None
        for cand in candidates:
            if cand.exists():
                s_dir = cand
                break
        if s_dir is None:
            continue
        dcm_files = sorted(list(s_dir.glob("*.dcm")))
        if not dcm_files:
            continue
        # Disambiguation: Keep the series with the higher slice count if duplicated
        if slot not in best_series_by_slot or len(dcm_files) > len(best_series_by_slot[slot]["files"]):
            best_series_by_slot[slot] = {
                "series_uid": series_uid,
                "files": dcm_files,
                "plane_code": slot.split("_")[0]
            }

    is_right_knee = False
    for slot, s_info in best_series_by_slot.items():
        geom = parse_series_geometry(s_info["files"])
        if geom:
            s_info["geom"] = geom
            if geom["is_right_knee"]:
                is_right_knee = True

    n_slots = len(slots)
    max_k = max([slice_budget.get(s.split("_")[0], 20) for s in slots]) if slots else 20
    vols = np.zeros((n_slots, max_k, image_size, image_size), dtype=np.uint8)
    mask = np.zeros(n_slots, dtype=bool)

    for slot_idx, slot in enumerate(slots):
        plane_code = slot.split("_")[0]
        k_depth = slice_budget.get(plane_code, max_k)
        if slot in best_series_by_slot:
            s_info = best_series_by_slot[slot]
            geom = s_info.get("geom") or parse_series_geometry(s_info["files"])
            if geom:
                slot_vol = process_series_to_volume(
                    geom["files"],
                    geom["spacing"],
                    plane_code,
                    target_depth=k_depth,
                    depth_band=depth_band,
                    is_right_knee=is_right_knee,
                    image_size=image_size
                )
                vols[slot_idx, :k_depth] = slot_vol
                mask[slot_idx] = True

    return vols, mask


def build_2p5d_slabs(
    vols: np.ndarray,
    mask: np.ndarray,
    k_budget: int = 20,
    depth_band: Tuple[float, float] = (0.04, 0.96)
) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs 2.5D tri-slice slabs [s_{t-1}, s_t, s_{t+1}] and normalized depth coordinates.

    Parameters
    ----------
    vols : np.ndarray
        (n_slots, max_slices, H, W) volumetric array in uint8.
    mask : np.ndarray
        (n_slots,) boolean array indicating active canonical slots.
    k_budget : int, default=20
        Number of slabs to sample along the volume depth.
    depth_band : Tuple[float, float], default=(0.04, 0.96)
        Normalized depth range [lo, hi] for slab sampling.

    Returns
    -------
    slabs : np.ndarray
        (n_slots, k_budget, 3, H, W) uint8 array of 3 adjacent slices per depth position.
    pos : np.ndarray
        (n_slots, k_budget) float32 array of normalized depth coordinates in [0, 1].
    """
    n_slots = vols.shape[0]
    H, W = vols.shape[2], vols.shape[3]
    slabs = np.zeros((n_slots, k_budget, 3, H, W), dtype=np.uint8)
    pos = np.zeros((n_slots, k_budget), dtype=np.float32)

    for slot in range(n_slots):
        if mask[slot]:
            n_avail = vols.shape[1]
            lo = max(1.0, depth_band[0] * (n_avail - 1))
            hi = min(n_avail - 2.0, depth_band[1] * (n_avail - 1))
            c_idx = np.clip(np.round(np.linspace(lo, hi, k_budget)).astype(int), 1, max(1, n_avail - 2))
            pos[slot] = c_idx / max(1.0, float(n_avail - 1))
            slabs[slot, :, 0] = vols[slot, np.clip(c_idx - 1, 0, n_avail - 1)]
            slabs[slot, :, 1] = vols[slot, c_idx]
            slabs[slot, :, 2] = vols[slot, np.clip(c_idx + 1, 0, n_avail - 1)]

    return slabs, pos

