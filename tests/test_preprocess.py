# tests/test_preprocess.py

# import os

import numpy as np
import pandas as pd
import pytest

from recruitment_fairness.data.preprocess import (
    encode_phases,
    save_npz,
    split_trials,
    vectorize_conditions,
)


def make_status_df(n_per_class=50):
    """Helper: create a DataFrame with balanced 'OverallStatus' classes."""
    df = pd.DataFrame({"OverallStatus": ["A"] * n_per_class + ["B"] * n_per_class})
    return df


def test_split_trials_stratified():
    df = make_status_df(n_per_class=50)
    train, val, test = split_trials(df, test_size=0.2, val_size=0.1, random_state=0)

    # sizes
    assert len(test) == 20
    assert len(val) == 10
    assert len(train) == 70

    # stratification preserved
    vc_train = train["OverallStatus"].value_counts(normalize=True)
    vc_val = val["OverallStatus"].value_counts(normalize=True)
    vc_test = test["OverallStatus"].value_counts(normalize=True)

    # each split should be ~50/50
    for vc in (vc_train, vc_val, vc_test):
        assert pytest.approx(vc["A"], rel=1e-2) == vc["B"]


def test_vectorize_conditions_shapes_and_vocab():
    train = pd.Series(["alpha beta", "beta gamma", "gamma delta"])
    val = pd.Series(["alpha beta"])
    test = pd.Series(["delta epsilon"])

    X_tr, X_va, X_te, vec = vectorize_conditions(train, val, test, max_features=10)

    # shapes
    assert X_tr.shape == (3, len(vec.vocabulary_))
    assert X_va.shape == (1, len(vec.vocabulary_))
    assert X_te.shape == (1, len(vec.vocabulary_))

    # vocabulary contains expected tokens
    for token in ["alpha", "beta", "gamma", "delta"]:
        assert token in vec.vocabulary_


def test_encode_phases_onehot():
    train = pd.Series(["Phase 1", "Phase 2", "Phase 3", "Phase 1"])
    val = pd.Series(["Phase 2"])
    test = pd.Series(["Phase 3"])

    X_tr, X_va, X_te, enc = encode_phases(train, val, test, drop="first")

    # drop='first' means 3 categories → 2 output columns
    assert X_tr.shape == (4, 2)
    assert X_va.shape == (1, 2)
    assert X_te.shape == (1, 2)

    # Check that val encoding matches the second category ("Phase 2")
    # Feature names are like ["Phase_Phase 2","Phase_Phase 3"]
    feat_names = enc.get_feature_names_out(["Phase"])
    assert feat_names.tolist() == ["Phase_Phase 2", "Phase_Phase 3"]
    assert X_va[0].tolist() == [1, 0]  # Phase 2 → [1,0]


def test_save_npz_creates_files_and_content(tmp_path):
    # create tiny arrays
    X_dict = {
        "train": np.arange(6).reshape(3, 2),
        "val": np.arange(4).reshape(2, 2),
        "test": np.arange(2).reshape(1, 2),
    }
    y_dict = {
        "train": np.array([0, 1, 0]),
        "val": np.array([1, 0]),
        "test": np.array([0]),
    }

    out_dir = tmp_path / "processed"
    save_npz(str(out_dir), X_dict, y_dict)

    # check files exist
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.npz"
        assert path.exists()

        # check contents
        data = np.load(path)
        assert np.array_equal(data["X"], X_dict[split])
        assert np.array_equal(data["y"], y_dict[split])
