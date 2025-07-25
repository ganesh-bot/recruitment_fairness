import numpy as np
import pandas as pd
import pytest

from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor


@pytest.fixture
def mock_df():
    # 10 rows, 5 with 'Completed' (is_success=1), 5 with other statuses (is_success=0)
    return pd.DataFrame(
        {
            "overall_status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Terminated",
                "Withdrawn",
                "Suspended",
                "Terminated",
                "Withdrawn",
            ],
            "sponsor_class": [
                "Industry",
                "NIH",
                "Other",
                "NIH",
                "Other",
                "Industry",
                "NIH",
                "Other",
                "Industry",
                "NIH",
            ],
            "interventions_names": [
                "DrugA",
                "DrugB",
                "DeviceA",
                "DeviceB",
                "DeviceC",
                "DrugC",
                "DrugD",
                "DeviceX",
                "DrugE",
                "DrugF",
            ],
            "phases": [
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Phase 1",
                "Phase 2",
                "Phase 1",
                "Phase 3",
            ],
            "enrollment_count": [100, 200, 150, 120, 130, 80, 90, 60, 70, 110],
        }
    )


def test_preprocess_basic(tmp_path, mock_df):
    pre = ClinicalTrialPreprocessor(data_dir=tmp_path, processed_dir=tmp_path)
    train, val, test = pre.preprocess(mock_df)

    # Check splits
    total = len(train) + len(val) + len(test)
    assert total == len(mock_df)

    # Check new column added
    for df in (train, val, test):
        assert "is_success" in df.columns
        assert set(df["is_success"].unique()).issubset({0, 1})

    # Check cleaned values
    assert train["sponsor_class"].isnull().sum() == 0
    assert train["interventions_names"].isnull().sum() == 0
    assert train["phases"].isnull().sum() == 0

    # Check files saved
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_encode_features(mock_df):
    pre = ClinicalTrialPreprocessor()
    encoded_X, encoded_y, ohe, le = pre.encode_features(mock_df)

    assert isinstance(encoded_X, np.ndarray)
    assert isinstance(encoded_y, np.ndarray)
    assert encoded_X.shape[0] == len(mock_df)
    assert encoded_y.shape[0] == len(mock_df)
    assert hasattr(ohe, "categories_")
    assert hasattr(le, "classes_")


def test_get_structured_features(mock_df):
    pre = ClinicalTrialPreprocessor()
    X, cat_cols = pre.get_structured_features(mock_df)

    assert isinstance(X, pd.DataFrame)
    assert "enrollment_count" in X.columns
    assert any(col.startswith("phase_") for col in X.columns)
    assert "sponsor_class" in X.columns
    assert isinstance(cat_cols, list)
    assert X.columns[cat_cols[0]] == "sponsor_class"
