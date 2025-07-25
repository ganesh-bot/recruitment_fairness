# src/recruitment_fairness/data/preprocess.py

# import json
from pathlib import Path

# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class ClinicalTrialPreprocessor:
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_raw(self):
        files = sorted(self.data_dir.glob("raw_clinical_trials_*.csv"))
        if not files:
            raise FileNotFoundError("No raw files found.")
        file = files[-1]
        df = pd.read_csv(file)
        print(f"📂 Loaded: {file}")
        return df

    def preprocess(self, df: pd.DataFrame):
        # Drop NAs for target/outcome
        df = df.dropna(subset=["overall_status"])

        # Use only sponsor_class (low cardinality), drop raw sponsor_name
        # drop any raw 'sponsor' or 'sponsor_name' columns if they exist
        df = df.drop(
            columns=[c for c in ["sponsor", "sponsor_name"] if c in df.columns]
        )

        # ensure sponsor_class is string and no NaNs
        df["sponsor_class"] = df["sponsor_class"].fillna("OTHER").astype(str)

        # Basic cleaning for intervention_names
        df["interventions_names"] = (
            df["interventions_names"].fillna("unknown").str.lower()
        )
        # Basic phase cleaning (collapse missing/NA to "unknown")
        df["phases"] = df["phases"].replace("", "unknown").fillna("unknown")
        # Optional: Add a "success" flag for the primary ML task
        df["is_success"] = df["overall_status"].apply(
            lambda x: 1 if x.lower() == "completed" else 0
        )
        # Split
        train, test = train_test_split(
            df, test_size=0.2, stratify=df["is_success"], random_state=42
        )
        train, val = train_test_split(
            train, test_size=0.1, stratify=train["is_success"], random_state=42
        )
        print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
        # Save splits
        train.to_csv(self.processed_dir / "train.csv", index=False)
        val.to_csv(self.processed_dir / "val.csv", index=False)
        test.to_csv(self.processed_dir / "test.csv", index=False)
        return train, val, test

    def encode_features(self, df: pd.DataFrame):
        # Example: One-hot encode 'phases', Label encode sponsor type
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        phase_encoded = ohe.fit_transform(df[["phases"]])
        le = LabelEncoder()
        status_encoded = le.fit_transform(df["overall_status"])
        # Output features/labels as arrays
        return phase_encoded, status_encoded, ohe, le

    def get_structured_features(self, df: pd.DataFrame):
        # 1) one-hot your phases (numeric)
        phases = pd.get_dummies(
            df["phases"].fillna("unknown").astype(str), prefix="phase"
        )
        # 2) keep sponsor_class raw (string)
        sponsor = df["sponsor_class"].astype(str).to_frame("sponsor_class")
        # 3) numeric enrollment
        enrollment = (
            df["enrollment_count"].fillna(0).astype(float).to_frame("enrollment_count")
        )

        # 4) concat
        X = pd.concat([phases, sponsor, enrollment], axis=1)

        # 5) CatBoost only needs the index of sponsor_class
        cat_cols = [X.columns.get_loc("sponsor_class")]

        return X, cat_cols
