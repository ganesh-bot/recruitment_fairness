# src/recruitment_fairness/data/preprocess.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .labels import label_outcome, label_recruitment_success


class ClinicalTrialPreprocessor:
    def __init__(
        self,
        data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def load_latest_raw(self) -> pd.DataFrame:
        files = sorted(self.data_dir.glob("raw_clinical_trials_*.csv"))
        if not files:
            raise FileNotFoundError("No raw files found.")
        df = pd.read_csv(files[-1])
        print(f"📂 Loaded raw data: {files[-1].name}")
        return df

    def preprocess(self, df: pd.DataFrame):
        # backfill if missing
        if (
            "planned_enrollment" not in df.columns
            or df["planned_enrollment"].isna().all()
        ):
            df["planned_enrollment"] = df["enrollment_count"]
        if (
            "actual_enrollment" not in df.columns
            or df["actual_enrollment"].isna().all()
        ):
            df["actual_enrollment"] = df["enrollment_count"]

        # 1) Drop missing overall status
        df = df.dropna(subset=["overall_status"])

        # 2) Label outcomes & recruitment
        df["y_outcome"] = label_outcome(df)
        df["y_recruit"] = label_recruitment_success(df)
        df = df.dropna(subset=["y_outcome", "y_recruit"]).copy()
        if df.empty:
            raise ValueError(
                "No trials left after labeling. "
                "Did you pull from the AACT snapshot (with actual_enrollment etc.), "
                "or only the v2 JSON API? You need planned vs. actual enrollment "
                "and dates to label."
            )

        # 3) Normalize some text columns
        df["sponsor_class"] = (
            df["sponsor_class"].fillna("OTHER").astype(str).str.upper()
        )
        df["phases"] = df["phases"].fillna("unknown").astype(str).str.lower()

        # 4) Pandemic flag
        df["year_started"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
        df["pandemic"] = df["year_started"].isin([2020, 2021]).astype(int)

        # ─── NEW STRUCTURED FEATURES ──────────────────────────────────────────
        # Enrollment rate (# per month)
        df["enroll_rate"] = (
            df["actual_enrollment"] / df["actual_duration_m"].replace({0: np.nan})
        ).fillna(0)

        # Enrollment ratio (actual vs planned)
        df["enroll_ratio"] = (
            df["actual_enrollment"] / df["planned_enrollment"].replace({0: np.nan})
        ).fillna(0)
        # ──────────────────────────────────────────────────────────────────────

        # ─── NEW COMBINED TEXT FIELD ──────────────────────────────────────────
        def clean_whitespace(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip()

        df["combined_text"] = (
            df["brief_title"].fillna("")
            + "  "
            + df["official_title"].fillna("")
            + "  "
            + df["brief_summary"].fillna("")
            + "  "
            + df["detailed_description"].fillna("")
            + "  "
            + df.get("eligibility_criteria", "").fillna("")
        ).apply(clean_whitespace)
        # ──────────────────────────────────────────────────────────────────────

        # 5) Train / Val / Test split (80/10/10 stratified on y_outcome)
        try:
            trainval, test = train_test_split(
                df,
                test_size=0.20,
                stratify=df["y_outcome"],
                random_state=self.random_state,
            )
            train, val = train_test_split(
                trainval,
                test_size=0.125,
                stratify=trainval["y_outcome"],
                random_state=self.random_state,
            )
        except ValueError as e:
            print(f"⚠️ Stratified split failed ({e}) — falling back to random split")
            trainval, test = train_test_split(
                df, test_size=0.20, random_state=self.random_state
            )
            train, val = train_test_split(
                trainval, test_size=0.125, random_state=self.random_state
            )

        print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

        # 6) Persist
        train.to_csv(self.processed_dir / "train.csv", index=False)
        val.to_csv(self.processed_dir / "val.csv", index=False)
        test.to_csv(self.processed_dir / "test.csv", index=False)

        return train, val, test

    def get_structured_features(self, df: pd.DataFrame):
        # force any blank or missing 'phases' into "unknown"
        phase_ser = (
            df["phases"]
            .fillna("unknown")  # catch NaNs
            .astype(str)
            .replace("", "unknown")  # catch literal empty strings
            .str.lower()
        )
        # now get_dummies will always produce a 'phase_unknown' and never bare 'phase_'
        phases = pd.get_dummies(phase_ser, prefix="phase")
        # (optional) sort your dummy columns for consistency:
        phases = phases.reindex(sorted(phases.columns), axis=1)

        # keep sponsor_class as single categorical (CatBoost index later)
        sponsor = df["sponsor_class"].astype(str).to_frame("sponsor_class")

        # numeric features — now includes the two new ones
        numeric_cols = [
            "enrollment_count",
            "planned_enrollment",
            "actual_enrollment",
            "planned_duration_m",
            "actual_duration_m",
            "num_arms",
            "has_dmc",
            "multi_country",
            "pandemic",
            "enroll_rate",
            "enroll_ratio",
        ]
        numeric = df[numeric_cols].fillna(0).astype(float)

        # assemble X
        X = pd.concat([phases, sponsor, numeric], axis=1)

        # find index of the single categorical column
        cat_idx = [X.columns.get_loc("sponsor_class")]

        return X, cat_idx
