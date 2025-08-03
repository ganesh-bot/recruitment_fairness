import numpy as np
from sklearn.metrics import roc_auc_score


def compute_group_metrics(y_true, y_pred, group_arr, y_proba=None, min_group_size=5):
    """
    Compute per-group TPR, FPR, AUC, positive rate, and deltas, skipping tiny groups.

    Args:
        y_true: array-like binary true labels (0/1)
        y_pred: array-like binary predictions (0/1)
        group_arr: array-like group membership (categorical)
        y_proba: array-like score/probability for AUC (same length as y_true)
        min_group_size: skip groups with fewer than this many examples or without both classes

    Returns:
        {
            "per_group": {group: {"TPR":..., "FPR":..., "AUC":..., "P_rate":...}, ...},
            "ΔTPR": float,
            "ΔFPR": float,
            "ΔAUC": float or None,
            "ΔP": float
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_arr = np.asarray(group_arr)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    per_group = {}
    tprs, fprs, aucs, p_rates = [], [], [], []

    unique_groups = np.unique(group_arr)
    for group in unique_groups:
        mask = group_arr == group
        yt = y_true[mask]
        yp = y_pred[mask]
        yp_proba = y_proba[mask] if y_proba is not None else None

        # skip tiny groups
        if len(yt) < min_group_size:
            continue
        if (yt == 1).sum() < 2 or (yt == 0).sum() < 2:
            continue

        # compute
        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        p_rate = np.mean(yp == 1)
        auc = (
            roc_auc_score(yt, yp_proba)
            if yp_proba is not None and len(np.unique(yt)) > 1
            else float("nan")
        )

        per_group[str(group)] = {
            "TPR": round(tpr, 3),
            "FPR": round(fpr, 3),
            "AUC": round(auc, 3) if not np.isnan(auc) else None,
            "P_rate": round(p_rate, 3),
        }

        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(auc if not np.isnan(auc) else 0.0)
        p_rates.append(p_rate)

    delta_tpr = round(max(tprs) - min(tprs), 3) if tprs else None
    delta_fpr = round(max(fprs) - min(fprs), 3) if fprs else None
    delta_auc = round(max(aucs) - min(aucs), 3) if y_proba is not None and aucs else None
    delta_p = round(max(p_rates) - min(p_rates), 3) if p_rates else None

    return {
        "per_group": per_group,
        "ΔTPR": delta_tpr,
        "ΔFPR": delta_fpr,
        "ΔAUC": delta_auc,
        "ΔP": delta_p,
    }
