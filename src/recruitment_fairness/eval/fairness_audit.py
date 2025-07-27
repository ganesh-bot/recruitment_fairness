from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def audit_groups(y_true, y_pred, group_labels, y_proba=None):
    """
    Compute TPR, FPR, AUC, and positive rate per group.
    Returns:
        report: dict containing metrics and ΔTPR, ΔFPR, ΔAUC, ΔP
    """
    group_metrics = defaultdict(dict)
    unique_groups = np.unique(group_labels)
    tprs, fprs, aucs, p_rates = [], [], [], []

    for group in unique_groups:
        idx = group_labels == group
        yt, yp = y_true[idx], y_pred[idx]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        p_rate = np.mean(yp)

        auc = (
            roc_auc_score(yt, y_proba[idx])
            if y_proba is not None and len(np.unique(yt)) > 1
            else float("nan")
        )

        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(auc)
        p_rates.append(p_rate)

        group_metrics[group] = {
            "TPR": round(tpr, 3),
            "FPR": round(fpr, 3),
            "AUC": round(auc, 3) if not np.isnan(auc) else None,
            "P_rate": round(p_rate, 3),
        }

    report = {
        "per_group": dict(group_metrics),
        "ΔTPR": round(max(tprs) - min(tprs), 3),
        "ΔFPR": round(max(fprs) - min(fprs), 3),
        "ΔAUC": round(max(aucs) - min(aucs), 3) if y_proba is not None else None,
        "ΔP": round(max(p_rates) - min(p_rates), 3),
    }
    return report
