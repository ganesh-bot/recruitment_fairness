# src/recruitment_fairness/evaluate.py

import argparse
import os

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from recruitment_fairness.data.clinicalbert_embedder import ClinicalBERTEmbedder
from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor
from recruitment_fairness.models.fairness_utils import (
    demographic_parity_difference,
    equal_opportunity_difference,
)


def main(args):
    # 1) Load processed splits
    train = pd.read_csv(os.path.join(args.data_processed, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_processed, "test.csv"))

    # 2) Structured features
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw,
        processed_dir=args.data_processed,
    )
    X_tr_struct, _ = preproc.get_structured_features(train)
    X_te_struct, _ = preproc.get_structured_features(test)

    # 3) ClinicalBERT embeddings for train & test
    embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
    texts_tr = train["brief_summary"].fillna("").astype(str).tolist()
    texts_te = test["brief_summary"].fillna("").astype(str).tolist()
    X_tr_text = embedder.embed_texts(
        texts_tr, batch_size=args.batch_size, max_length=args.max_length
    )
    X_te_text = embedder.embed_texts(
        texts_te, batch_size=args.batch_size, max_length=args.max_length
    )

    # 4) Assemble DataFrames & align columns
    text_cols = [f"text_{i}" for i in range(X_tr_text.shape[1])]
    df_tr_text = pd.DataFrame(X_tr_text, columns=text_cols, index=X_tr_struct.index)
    df_te_text = pd.DataFrame(X_te_text, columns=text_cols, index=X_te_struct.index)

    X_tr_df = pd.concat([X_tr_struct, df_tr_text], axis=1)
    X_te_df = pd.concat([X_te_struct, df_te_text], axis=1)

    # Align test to train columns
    X_te_df = X_te_df.reindex(columns=X_tr_df.columns, fill_value=0)

    y_tr = train["is_success"].to_numpy()
    y_test = test["is_success"].to_numpy()

    # 5) Load trained models
    recruit_model = CatBoostClassifier()
    recruit_model.load_model(os.path.join(args.model_dir, "recruitment.cbm"))

    fair_model = CatBoostClassifier()
    fair_model.load_model(os.path.join(args.model_dir, "fair_outcome.cbm"))

    # 6) Predict RecruitmentNet
    train_pool = Pool(data=X_tr_df, label=y_tr, cat_features=["sponsor_class"])
    test_pool = Pool(data=X_te_df, label=y_test, cat_features=["sponsor_class"])
    y_pred_rec = recruit_model.predict_proba(test_pool)[:, 1]

    # 7) RecruitmentNet metrics
    auc_rec = roc_auc_score(y_test, y_pred_rec)
    f1_rec = f1_score(y_test, y_pred_rec > 0.5)
    acc_rec = accuracy_score(y_test, y_pred_rec > 0.5)
    cm_rec = confusion_matrix(y_test, y_pred_rec > 0.5)

    print("\n=== RecruitmentNet Performance ===")
    print(f"AUC: {auc_rec:.3f} | F1: {f1_rec:.3f} | Acc: {acc_rec:.3f}")
    print("Confusion Matrix:\n", cm_rec)
    print(classification_report(y_test, y_pred_rec > 0.5))

    # 8) Build second-stage features (add recruit_score)
    recruit_score_tr = recruit_model.predict_proba(train_pool)[:, 1].reshape(-1, 1)
    recruit_score_te = y_pred_rec.reshape(-1, 1)

    df2_tr = pd.concat(
        [
            X_tr_df,
            pd.DataFrame(
                recruit_score_tr, columns=["recruit_score"], index=X_tr_df.index
            ),
        ],
        axis=1,
    )
    df2_te = pd.concat(
        [
            X_te_df,
            pd.DataFrame(
                recruit_score_te, columns=["recruit_score"], index=X_te_df.index
            ),
        ],
        axis=1,
    )
    # align test2 to train2
    df2_te = df2_te.reindex(columns=df2_tr.columns, fill_value=0)

    # 9) Predict FairOutcomeNet
    test2_pool = Pool(data=df2_te, label=y_test, cat_features=["sponsor_class"])
    y_pred_out = fair_model.predict_proba(test2_pool)[:, 1]

    # 10) FairOutcomeNet metrics
    auc_out = roc_auc_score(y_test, y_pred_out)
    f1_out = f1_score(y_test, y_pred_out > 0.5)
    acc_out = accuracy_score(y_test, y_pred_out > 0.5)
    cm_out = confusion_matrix(y_test, y_pred_out > 0.5)

    print("\n=== FairOutcomeNet Performance ===")
    print(f"AUC: {auc_out:.3f} | F1: {f1_out:.3f} | Acc: {acc_out:.3f}")
    print("Confusion Matrix:\n", cm_out)
    print(classification_report(y_test, y_pred_out > 0.5))

    # 11) ROC Curves
    fpr_r, tpr_r, _ = roc_curve(y_test, y_pred_rec)
    fpr_o, tpr_o, _ = roc_curve(y_test, y_pred_out)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_r, tpr_r, label=f"RecruitmentNet (AUC={auc_rec:.2f})")
    plt.plot(fpr_o, tpr_o, label=f"FairOutcomeNet (AUC={auc_out:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 12) Fairness audit
    sens = test["sponsor_class"].fillna("unknown").to_numpy()
    dp_r = demographic_parity_difference(y_test, y_pred_rec > 0.5, sens)
    eo_r = equal_opportunity_difference(y_test, y_pred_rec > 0.5, sens)
    dp_o = demographic_parity_difference(y_test, y_pred_out > 0.5, sens)
    eo_o = equal_opportunity_difference(y_test, y_pred_out > 0.5, sens)

    print(f"\nFairness ΔP (Recruit): {dp_r:.3f}, ΔTPR: {eo_r:.3f}")
    print(f"Fairness ΔP (Outcome): {dp_o:.3f}, ΔTPR: {eo_o:.3f}")

    # ─────────────────────────────────────────────────────────────────
    # After printing metrics, just BEFORE exiting main()
    # ─────────────────────────────────────────────────────────────────

    # 0) Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── 1) Save ROC Curves ──────────────────────────────────────────
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_r, tpr_r, label=f"RecruitmentNet (AUC={auc_rec:.2f})")
    plt.plot(fpr_o, tpr_o, label=f"FairOutcomeNet (AUC={auc_out:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved ROC curves to {roc_path}")

    # ─── 2) Save SHAP Summary ───────────────────────────────────────
    explainer = shap.TreeExplainer(fair_model)
    sample_df = df2_te.sample(n=args.shap_samples, random_state=args.seed)
    sample_pool = Pool(data=sample_df, cat_features=["sponsor_class"])
    shap_vals = explainer.shap_values(sample_pool)

    shap.summary_plot(shap_vals, sample_df, show=False)
    shap_path = os.path.join(args.output_dir, "shap_summary.png")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved SHAP summary to {shap_path}")

    # ─── 3) Save two sample SHAP Waterfall Plots ───────────────────
    # pick first two samples from that same sample_df
    for i, idx in enumerate(sample_df.index[:2]):
        # compute SHAP for this single row
        sv_single = explainer.shap_values(
            Pool(data=sample_df.loc[[idx]], cat_features=["sponsor_class"])
        )
        # build an Explanation object
        exp = shap.Explanation(
            values=sv_single[0],
            base_values=explainer.expected_value,
            data=sample_df.loc[idx],
            feature_names=sample_df.columns,
        )
        shap.plots.waterfall(exp, max_display=10, show=False)
        wf_path = os.path.join(args.output_dir, f"shap_waterfall_{idx}.png")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig(wf_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved SHAP waterfall for sample {idx} to {wf_path}")

    # ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_raw", default="data/raw")
    p.add_argument("--data_processed", default="data/processed")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--output_dir", default="results")  # new
    p.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--shap_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
