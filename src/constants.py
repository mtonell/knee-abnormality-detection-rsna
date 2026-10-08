"""Canonical constants, target findings, and physical MRI geometry definitions."""

from typing import List, Dict, Tuple

# -------------------------------------------------------------------------
# 12 Target Knee Abnormalities (Official RSNA Competition Order)
# -------------------------------------------------------------------------
TARGET_COLS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]
NUM_CLASSES: int = len(TARGET_COLS)

# -------------------------------------------------------------------------
# Anatomical Planes & Canonical Series Slots
# -------------------------------------------------------------------------
PLANE_MAP: Dict[str, str] = {
    "sagittal": "SAG",
    "coronal": "COR",
    "axial": "AX",
    "sag": "SAG",
    "cor": "COR",
    "ax": "AX"
}

# 6 Canonical Slots (ConvNeXt-Nano 6-View Multi-Head Architecture)
CONVNEXT_SERIES_SLOTS: List[str] = [
    "SAG_N",   # Sagittal Non-Fat-Suppressed
    "SAG_FS",  # Sagittal Fat-Suppressed
    "COR_N",   # Coronal Non-Fat-Suppressed
    "COR_FS",  # Coronal Fat-Suppressed
    "AX_N",    # Axial Non-Fat-Suppressed
    "AX_FS"    # Axial Fat-Suppressed
]

# 4 Diverse Slots (RadImageNet ResNet-50 E11 Architecture)
RADIMAGENET_SERIES_SLOTS: List[str] = [
    "SAG_FS",  # Sagittal Fat-Suppressed
    "COR_FS",  # Coronal Fat-Suppressed
    "AX_FS",   # Axial Fat-Suppressed
    "SAG_N"    # Non-FS anatomical reference anchor
]

RADIMAGENET_SLOTS: List[Tuple[str, str, bool]] = [
    ("SAG_FS", "Sagittal", True),
    ("COR_FS", "Coronal", True),
    ("AX_FS", "Axial", True),
    ("SAG_NOFS", "Sagittal", False)   # Non-FS anatomical reference anchor
]

# -------------------------------------------------------------------------
# Physical Geometry: View-Dependent Millimeter Square Crops & Anchors
# -------------------------------------------------------------------------
# Centered on joint tissue to preserve cruciate notch and collateral margins
PLANE_SQUARE_CROP_MM: Dict[str, float] = {
    "SAG": 135.0,  # Sagittal: 135mm (Cruciate notch, meniscal horns, patellar tendon)
    "COR": 150.0,  # Coronal: 150mm (Femoral condyles, tibial plateaus, collaterals)
    "AX": 120.0    # Axial: 120mm (Patellofemoral joint, trochlear groove)
}

PLANE_Y_ANCHOR: Dict[str, float] = {
    "SAG": 0.52,
    "COR": 0.52,
    "AX": 0.48
}

DEFAULT_IMAGE_SIZE: int = 224


# Calibrated Clinical Risk Stratification Thresholds
# They were determined using the following criteria:
# yellow -> thr where at least 95% of positive cases are considered positive
# orange -> thr that maximize F1
# red -> thr where 95% or positives are actual positive cases
# Threshold were determined with gold labels, so not reliable for a general purpose (tune with other larger validation step)
CLINICAL_THRESHOLDS = {
    "ACL": {"yellow": 0.50, "orange": 0.54, "red": 0.62},
    "MCL": {"yellow": 0.67, "orange": 0.78, "red": 0.92},
    "Medial Meniscus": {"yellow": 0.41, "orange": 0.53, "red": 0.59},
    "Lateral Meniscus": {"yellow": 0.33, "orange": 0.46, "red": 0.81},
    "Medial OA": {"yellow": 0.59, "orange": 0.82, "red": 0.87},
    "Lateral OA": {"yellow": 0.37, "orange": 0.77, "red": 0.89},
    "PF OA": {"yellow": 0.25, "orange": 0.64, "red": 0.74},
    "Effusion": {"yellow": 0.35, "orange": 0.40, "red": 0.44},
    "Synovitis": {"yellow": 0.28, "orange": 0.41, "red": 0.81},
    "Baker's": {"yellow": 0.66, "orange": 0.75, "red": 0.80},
    "Contusion": {"yellow": 0.55, "orange": 0.59, "red": 0.80},
    "Fracture": {"yellow": 0.40, "orange": 0.60, "red": 0.71},
}
