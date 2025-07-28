# src/recruitment_fairness/data/labels.py

import numpy as np
import pandas as pd


def label_outcome(df: pd.DataFrame) -> pd.Series:
    """1 if Completed; 0 if Terminated/Withdrawn/Suspended; else NaN."""
    status = df["overall_status"].str.lower().fillna("")
    completed = status == "completed"
    failed = status.isin(["terminated", "withdrawn", "suspended"])
    label = pd.Series(np.nan, index=df.index)
    label[completed] = 1.0
    label[failed] = 0.0
    return label.rename("y_outcome")


def label_recruitment_success(df: pd.DataFrame, grace_months: int = 3) -> pd.Series:
    """
    Proxy label for recruitment success using enrollment rate:
    1 if actual_enrollment / actual_duration_m >= median rate
    0 otherwise (if both fields exist), NaN if missing.

    Old rule (commented for reference):
    1 if actual_enrollment >= 0.8 * planned_enrollment
           AND actual_duration_m <= planned_duration_m + grace_months
    """
    # Check required fields
    valid = df["actual_enrollment"].notna() & df["actual_duration_m"].notna()
    label = pd.Series(np.nan, index=df.index)

    # Compute enrollment rate
    df = df.copy()
    df["enroll_rate"] = df["actual_enrollment"] / df["actual_duration_m"]

    # Use median threshold
    rate_threshold = df.loc[valid, "enroll_rate"].median()

    # Apply new rule
    label[valid] = (df.loc[valid, "enroll_rate"] >= rate_threshold).astype(float)

    # --- (Optional) Old rule for reference ---
    # mask = (
    #     df["planned_enrollment"].notna()
    #     & df["actual_enrollment"].notna()
    #     & df["planned_duration_m"].notna()
    #     & df["actual_duration_m"].notna()
    # )
    # cond_success = (df["actual_enrollment"] >= 0.8 * df["planned_enrollment"]) & (
    #     df["actual_duration_m"] <= df["planned_duration_m"] + grace_months
    # )
    # label[mask] = np.where(cond_success[mask], 1.0, 0.0)

    return label.rename("y_recruit")
