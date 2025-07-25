# src/recruitment_fairness/models/fairness_utils.py

import numpy as np


def demographic_parity_difference(y_true, y_pred, sensitive_attr):
    # y_pred should be binary or probability (use threshold 0.5 if needed)
    groups = np.unique(sensitive_attr)
    rates = []
    for g in groups:
        idx = sensitive_attr == g
        rates.append(np.mean(y_pred[idx]))
    return np.abs(rates[0] - rates[1])  # If binary sensitive attribute


def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    tprs = []
    for g in groups:
        mask = (sensitive_attr == g) & (y_true == 1)
        if mask.sum() == 0:
            continue  # skip groups with no positives
        tprs.append(np.mean(y_pred[mask]))
    if len(tprs) < 2:
        return np.nan  # not enough groups to compare
    return abs(tprs[0] - tprs[1])
