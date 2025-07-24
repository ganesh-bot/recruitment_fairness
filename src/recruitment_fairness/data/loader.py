# src/recruitment_fairness/data/loader.py

import os
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
CACHE_FILE = DATA_DIR / "clinical_trials.csv"
API_URL = (
    "https://clinicaltrials.gov/api/query/study_fields"
    "?expr=&fields=NCTId,OverallStatus,Phase,Condition,InterventionName"
    "&min_rnk=1&max_rnk=1000&fmt=csv"
)


def download_trials(max_rank: int = 1000, force: bool = False) -> pd.DataFrame:
    """
    Download trial data from ClinicalTrials.gov API in CSV format and cache locally.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force:
        return pd.read_csv(CACHE_FILE)

    url = API_URL.replace("1000", str(max_rank))
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(resp.text))
    df.to_csv(CACHE_FILE, index=False)
    return df


def load_trials(force_download: bool = False) -> pd.DataFrame:
    """
    Wrapper that returns a cleaned DataFrame of trials, downloading if needed.
    """
    df = download_trials(force=force_download)
    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning: drop rows with missing NCTId or OverallStatus,
    standardize text columns, etc.
    """
    # 1. Drop rows with missing critical fields
    df = df.dropna(subset=["NCTId", "OverallStatus"])

    # 2. Make an explicit copy so we don’t warn on chained indexing
    df = df.copy()

    # 3. Standardize string columns using .loc to avoid SettingWithCopyWarning
    df.loc[:, "Condition"] = df["Condition"].str.lower().str.strip()
    df.loc[:, "InterventionName"] = df["InterventionName"].fillna("unknown").str.lower()
    return df


if __name__ == "__main__":
    # Quick smoke test
    data = load_trials()
    print(data.head())
