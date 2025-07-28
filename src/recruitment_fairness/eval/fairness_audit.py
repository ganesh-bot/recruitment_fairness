# from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def audit_groups(y_true, y_pred, group_labels, y_proba=None):
    """
    Compute TPR, FPR, AUC, and positive rate per group.
    group_labels: a dict of {column_name: group_array}
    Returns:
        report: dict with 'per_group' per column and ΔTPR/ΔFPR/ΔAUC/ΔP per column
    """
    report = {"per_group": {}, "ΔTPR": {}, "ΔFPR": {}, "ΔAUC": {}, "ΔP": {}}

    for group_col, group_arr in group_labels.items():
        metrics_by_group = {}
        unique_groups = np.unique(group_arr)
        tprs, fprs, aucs, p_rates = [], [], [], []

        for group in unique_groups:
            idx = group_arr == group
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

            metrics_by_group[str(group)] = {
                "TPR": round(tpr, 3),
                "FPR": round(fpr, 3),
                "AUC": round(auc, 3) if not np.isnan(auc) else None,
                "P_rate": round(p_rate, 3),
            }

            tprs.append(tpr)
            fprs.append(fpr)
            aucs.append(auc if not np.isnan(auc) else 0.0)
            p_rates.append(p_rate)

        report["per_group"][group_col] = metrics_by_group
        report["ΔTPR"][group_col] = round(max(tprs) - min(tprs), 3)
        report["ΔFPR"][group_col] = round(max(fprs) - min(fprs), 3)
        report["ΔAUC"][group_col] = (
            round(max(aucs) - min(aucs), 3) if y_proba is not None else None
        )
        report["ΔP"][group_col] = round(max(p_rates) - min(p_rates), 3)

    return report
