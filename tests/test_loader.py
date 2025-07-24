# tests/test_loader.py
import pandas as pd

from recruitment_fairness.data.loader import clean_dataframe


def make_raw_df():
    return pd.DataFrame(
        {
            "NCTId": ["NCT0001", None],
            "OverallStatus": ["Completed", "Recruiting"],
            "Phase": ["Phase 1", "Phase 2"],
            "Condition": [" Diabetes ", None],
            "InterventionName": [None, "DrugA"],
        }
    )


def test_clean_dataframe_drops_missing_ids():
    df = make_raw_df()
    cleaned = clean_dataframe(df)
    # Should drop the None NCTId row
    assert cleaned.shape[0] == 1


def test_clean_dataframe_standardizes_text():
    df = make_raw_df()
    cleaned = clean_dataframe(df)
    assert cleaned.loc[0, "Condition"] == "diabetes"
    assert cleaned.loc[0, "InterventionName"] == "unknown"
