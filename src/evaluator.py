"""Multilabel Macro ROC-AUC evaluation and detailed scorecard generator."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.constants import TARGET_COLS


def compute_roc_auc_single(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes binary ROC-AUC with pure numpy/scipy fallback if sklearn is absent."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true > 0.5, y_score))
    except ImportError:
        pos = y_score[y_true > 0.5]
        neg = y_score[y_true <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Mann-Whitney U rank statistic
        r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        u = np.sum(r[:len(pos)]) - len(pos) * (len(pos) + 1) / 2.0
        return float(u / (len(pos) * len(neg)))


def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = TARGET_COLS
) -> Tuple[float, Dict[str, float]]:
    """Computes unweighted macro-averaged ROC-AUC across all binary targets."""
    class_aucs: Dict[str, float] = {}
    valid_scores: List[float] = []

    for i, col in enumerate(target_names):
        true_col = y_true[:, i]
        if len(np.unique(true_col > 0.5)) > 1:
            auc = compute_roc_auc_single(true_col, y_pred[:, i])
            class_aucs[col] = float(auc)
            valid_scores.append(auc)
        else:
            class_aucs[col] = 0.5

    macro_auc = float(np.mean(valid_scores)) if valid_scores else 0.5
    return macro_auc, class_aucs


def format_scorecard_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = TARGET_COLS
) -> pd.DataFrame:
    """Generates a detailed per-finding scorecard table with positive counts and AUCs."""
    _, class_aucs = compute_macro_auc(y_true, y_pred, target_names)
    rows = []
    n_samples = len(y_true)

    for i, col in enumerate(target_names):
        pos_cnt = int(np.sum(y_true[:, i] > 0.5))
        prev = pos_cnt / max(1, n_samples)
        rows.append({
            "Pathology Finding": col,
            "Positive Count": pos_cnt,
            "Prevalence": f"{prev:.1%}",
            "ROC-AUC": f"{class_aucs.get(col, 0.5):.4f}"
        })

    df = pd.DataFrame(rows)
    macro_mean = np.mean([class_aucs[c] for c in target_names])
    summary_row = pd.DataFrame([{
        "Pathology Finding": "== MACRO AVERAGE ==",
        "Positive Count": int(np.sum(y_true > 0.5)),
        "Prevalence": "-",
        "ROC-AUC": f"{macro_mean:.4f}"
    }])
    return pd.concat([df, summary_row], ignore_index=True)
