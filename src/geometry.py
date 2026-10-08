"""3D DICOM geometry processing, slice depth sorting, square cropping, and laterality normalization."""

from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import cv2
import pydicom

from src.constants import PLANE_SQUARE_CROP_MM, PLANE_Y_ANCHOR, DEFAULT_IMAGE_SIZE


def parse_series_geometry(dicom_files: List[Path]) -> Optional[Dict]:
    """Sorts slices by projecting ImagePositionPatient onto the plane normal vector.

    Calculates plane normal vector: n = row x col.
    Slices are ordered by depth = dot(IPP, n) to ensure true physical anatomical progression.
    Detects knee laterality (Right vs. Left).
    """
    if not dicom_files:
        return None

    headers = []
    for f in dicom_files:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            headers.append((f, h))
        except Exception:
            continue

    if not headers:
        return None

    d0 = headers[0][1]
    iop = getattr(d0, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        headers.sort(key=lambda t: float(getattr(t[1], "InstanceNumber", 0) or 0))
        return {
            "files": [f for f, _ in headers],
            "spacing": (0.5, 0.5),
            "is_right_knee": False
        }

    row = np.array(iop[:3], dtype=np.float32)
    col = np.array(iop[3:], dtype=np.float32)
    normal = np.cross(row, col)
    norm_val = np.linalg.norm(normal)
    if norm_val > 0:
        normal = normal / norm_val

    def get_slice_depth(h) -> float:
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(np.dot(np.array(ipp, dtype=np.float32), normal))
        return float(getattr(h, "InstanceNumber", 0) or 0)

    headers.sort(key=lambda t: get_slice_depth(t[1]))

    ps = getattr(d0, "PixelSpacing", [0.5, 0.5])
    try:
        spacing = (float(ps[0]), float(ps[1]))
    except Exception:
        spacing = (0.5, 0.5)

    # 3-Stage Knee Laterality Detection (Tags -> Text Heuristics -> LPS Coordinates)
    lat_tag = str(
        getattr(d0, "Laterality", "") or
        getattr(d0, "ImageLaterality", "") or ""
    ).upper().strip()
    if lat_tag in ("R", "RIGHT"):
        is_right_knee = True
    elif lat_tag in ("L", "LEFT"):
        is_right_knee = False
    else:
        # Check SeriesDescription and ProtocolName for lateral cues
        desc = (str(getattr(d0, "SeriesDescription", "")) + " " + str(getattr(d0, "ProtocolName", ""))).upper()
        if any(tok in desc for tok in [" RT", "RIGHT", " R ", "_R_", "RT_"]):
            is_right_knee = True
        elif any(tok in desc for tok in [" LT", "LEFT", " L ", "_L_", "LT_"]):
            is_right_knee = False
        else:
            # Physical LPS coordinate fallback: in DICOM LPS, X < 0 mm corresponds to anatomical right
            cx_vals = [
                float(getattr(h, "ImagePositionPatient", [0])[0])
                for _, h in headers
                if getattr(h, "ImagePositionPatient", None) is not None and len(getattr(h, "ImagePositionPatient")) >= 1
            ]
            if cx_vals:
                is_right_knee = bool(np.median(cx_vals) < 0.0)
            else:
                is_right_knee = False

    return {
        "files": [f for f, _ in headers],
        "spacing": spacing,
        "is_right_knee": is_right_knee
    }


def crop_and_pad_square(
    img: np.ndarray,
    spacing: Tuple[float, float],
    plane_code: str,
    out_px: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Extracts a strictly square (1:1 aspect ratio) physical millimeter field-of-view crop.

    Uses view-dependent millimeter bounding boxes (SAG: 135mm, COR: 150mm, AX: 120mm)
    centered on joint tissue via 25th percentile foreground thresholding.
    """
    h, w = img.shape
    crop_mm = PLANE_SQUARE_CROP_MM.get(plane_code, 135.0)
    avg_spacing = (spacing[0] + spacing[1]) / 2.0
    box_size = int(round(crop_mm / avg_spacing))
    box_size = max(64, min(box_size, max(h, w) * 2))

    threshold = np.percentile(img[img > 0], 25) if np.any(img > 0) else 10
    tissue_mask = img > threshold
    if np.sum(tissue_mask) > 100:
        cy_tissue, cx_tissue = np.argwhere(tissue_mask).mean(axis=0)
        y_ratio = PLANE_Y_ANCHOR.get(plane_code, 0.52)
        cy = int(round(cy_tissue * 0.7 + (h * y_ratio) * 0.3))
        cx = int(round(cx_tissue * 0.8 + (w * 0.5) * 0.2))
    else:
        cy = int(round(h * PLANE_Y_ANCHOR.get(plane_code, 0.52)))
        cx = int(round(w * 0.5))

    half = box_size // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        padded = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=int(img.min())
        )
        cropped = padded[y1 + pad_top:y2 + pad_top, x1 + pad_left:x2 + pad_left]
    else:
        cropped = img[y1:y2, x1:x2]

    if cropped.shape[0] != out_px or cropped.shape[1] != out_px:
        cropped = cv2.resize(cropped, (out_px, out_px), interpolation=cv2.INTER_AREA)

    return cropped


def apply_laterality_normalization(
    volume: np.ndarray,
    plane_code: str,
    is_right_knee: bool
) -> np.ndarray:
    """Normalizes anatomical laterality to a canonical left-knee standard.

    Coronal & Axial right knees: horizontal reflection (flip width).
    Sagittal right knees: slice-stack depth reversal (volume[::-1, :, :]).
      -> Crucial: preserves the internal Anterior <-> Posterior slice alignment
         while standardizing lateral-to-medial progression!
    """
    if not is_right_knee:
        return volume

    if plane_code in ("COR", "AX", "Coronal", "Axial"):
        # Flip along horizontal axis (width)
        return np.ascontiguousarray(volume[:, :, ::-1])
    elif plane_code in ("SAG", "Sagittal"):
        # Reverse depth slice-stack order
        return np.ascontiguousarray(volume[::-1, :, :])

    return volume
