"""RSNA Knee Abnormality Detection: Dual-Architecture Deep Learning System."""

from src.loss import (
    SampleWeightedBCEWithLogitsLoss,
    SampleWeightedFocalLoss,
    get_loss_function
)
from src.xai import (
    resolve_xai_targets,
    extract_slice_attention,
    extract_multiview_slice_attention,
    compute_gradcam_slab,
    render_xai_card,
    render_multiview_attention_curve,
    render_top_slices_gradcam_grid
)
from src.ensembling import blend_predictions

__version__ = "1.0.0"

