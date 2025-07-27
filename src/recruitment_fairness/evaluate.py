# src/recruitment_fairness/evaluate.py

import argparse
import json
import os

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
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
from recruitment_fairness.eval.fairness_audit import audit_groups
from recruitment_fairness.models.fairness_utils import (
    demographic_parity_difference,
    equal_opportunity_difference,
)


def main(args):
    # 1) Load processed splits
    train = pd.read_csv(os.path.join(args.data_processed, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(args.data_processed, "test.csv"), index_col=0)

    # 2) Structured features + get cat indices
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw,
        processed_dir=args.data_processed,
        random_state=args.seed,
    )
    X_tr_struct, cat_idx = preproc.get_structured_features(train)
    X_te_struct, _ = preproc.get_structured_features(test)

    # 3) Embed texts with ClinicalBERT
    embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
    texts_tr = train["combined_text"].fillna("").astype(str).tolist()
    texts_te = test["combined_text"].fillna("").astype(str).tolist()

    X_tr_text = embedder.embed_texts(
        texts_tr, batch_size=args.batch_size, max_length=args.max_length
    )
    X_te_text = embedder.embed_texts(
        texts_te, batch_size=args.batch_size, max_length=args.max_length
    )

    # 4) PCA → reduce to 50 dims
    pca = PCA(n_components=50, random_state=args.seed)
    X_tr_txt = pca.fit_transform(X_tr_text)
    X_te_txt = pca.transform(X_te_text)

    # 5) Assemble full feature frames
    text_cols = [f"text_{i}" for i in range(X_tr_txt.shape[1])]
    df_tr_text = pd.DataFrame(X_tr_txt, index=X_tr_struct.index, columns=text_cols)
    df_te_text = pd.DataFrame(X_te_txt, index=X_te_struct.index, columns=text_cols)

    df_tr = pd.concat([X_tr_struct, df_tr_text], axis=1)
    df_te = pd.concat([X_te_struct, df_te_text], axis=1)
    df_te = df_te.reindex(columns=df_tr.columns, fill_value=0)

    # 6) True labels
    y_te_rec = test["y_recruit"].to_numpy()
    y_te_out = test["y_outcome"].to_numpy()

    # 7) Load models
    rec_model = CatBoostClassifier()
    rec_model.load_model(os.path.join(args.model_dir, "recruitment.cbm"))
    fair_model = CatBoostClassifier()
    fair_model.load_model(os.path.join(args.model_dir, "fair_outcome.cbm"))

    # 8) RecruitmentNet predictions & metrics
    pool_te = Pool(data=df_te, cat_features=cat_idx)
    y_pred_rec = rec_model.predict_proba(pool_te)[:, 1]

    auc_r = roc_auc_score(y_te_rec, y_pred_rec)
    f1_r = f1_score(y_te_rec, y_pred_rec > 0.5)
    acc_r = accuracy_score(y_te_rec, y_pred_rec > 0.5)
    cm_r = confusion_matrix(y_te_rec, y_pred_rec > 0.5)

    dp_r = demographic_parity_difference(
        y_te_rec, y_pred_rec > 0.5, test["sponsor_class"].to_numpy()
    )
    eo_r = equal_opportunity_difference(
        y_te_rec, y_pred_rec > 0.5, test["sponsor_class"].to_numpy()
    )

    print("\n=== RecruitmentNet ===")
    print(f"AUC   {auc_r:.3f} | F1  {f1_r:.3f} | Acc {acc_r:.3f}")
    print("Confusion Matrix:\n", cm_r)

    print(classification_report(y_te_rec, y_pred_rec > 0.5))

    print(f"Fairness ΔP={dp_r:.3f}, ΔTPR={eo_r:.3f}")
    for group_col in ["sponsor_class", "region_income_group", "therapeutic_area"]:
        if group_col in test.columns:
            fair_report_r = audit_groups(
                y_true=y_te_rec,
                y_pred=(y_te_rec > 0.5),
                group_labels=test[group_col],
                y_proba=y_te_rec,
            )
            print(
                f"[{group_col}] ΔTPR={fair_report_r['ΔTPR']:.3f}, "
                "ΔAUC={fair_report_r['ΔAUC']:.3f}, ΔP={fair_report_r['ΔP']:.3f}"
            )

    # 9) Build 2nd-stage feature frames
    rec_score_te = y_pred_rec.reshape(-1, 1)
    df2_tr = pd.concat(
        [
            df_tr,
            pd.DataFrame(
                rec_model.predict_proba(Pool(data=df_tr, cat_features=cat_idx))[
                    :, 1
                ].reshape(-1, 1),
                index=df_tr.index,
                columns=["recruit_score"],
            ),
        ],
        axis=1,
    )
    df2_te = pd.concat(
        [
            df_te,
            pd.DataFrame(rec_score_te, index=df_te.index, columns=["recruit_score"]),
        ],
        axis=1,
    )
    df2_te = df2_te.reindex(columns=df2_tr.columns, fill_value=0)

    # 10) FairOutcomeNet predictions & metrics
    pool2_te = Pool(data=df2_te, cat_features=cat_idx)
    y_pred_out = fair_model.predict_proba(pool2_te)[:, 1]

    auc_o = roc_auc_score(y_te_out, y_pred_out)
    f1_o = f1_score(y_te_out, y_pred_out > 0.5)
    acc_o = accuracy_score(y_te_out, y_pred_out > 0.5)
    cm_o = confusion_matrix(y_te_out, y_pred_out > 0.5)

    dp_o = demographic_parity_difference(
        y_te_out, y_pred_out > 0.5, test["sponsor_class"].to_numpy()
    )
    eo_o = equal_opportunity_difference(
        y_te_out, y_pred_out > 0.5, test["sponsor_class"].to_numpy()
    )

    print("\n=== FairOutcomeNet ===")
    print(f"AUC   {auc_o:.3f} | F1  {f1_o:.3f} | Acc {acc_o:.3f}")
    print("Confusion Matrix:\n", cm_o)
    print(classification_report(y_te_out, y_pred_out > 0.5))

    print(f"Fairness ΔP={dp_o:.3f}, ΔTPR={eo_o:.3f}")

    for group_col in ["sponsor_class", "region_income_group", "therapeutic_area"]:
        if group_col in test.columns:
            fair_report_o = audit_groups(
                y_true=y_te_out,  # or y_te_rec
                y_pred=(y_pred_out > 0.5),
                group_labels=test[group_col],
                y_proba=y_pred_out,
            )
            print(
                f"[{group_col}] ΔTPR={fair_report_o['ΔTPR']:.3f}, "
                "ΔAUC={fair_report_o['ΔAUC']:.3f}, ΔP={fair_report_o['ΔP']:.3f}"
            )

    with open(os.path.join(args.model_dir, "fairness_recruitment.json"), "w") as f:
        json.dump(fair_report_r, f, indent=2)
    with open(os.path.join(args.model_dir, "fairness_outcome.json"), "w") as f:
        json.dump(fair_report_o, f, indent=2)

    def plot_fairness_metric(metric_dict, title, save_path):
        groups = list(metric_dict.keys())
        values = [metric_dict[g] for g in groups]
        plt.figure(figsize=(8, 4))
        plt.bar(groups, values)
        plt.xticks(rotation=45)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    plot_fairness_metric(
        {k: v["P_rate"] for k, v in fair_report_o["per_group"].items()},
        "FairOutcomeNet P_rate by Therapeutic Area",
        "results/p_rate_therapeutic_area.png",
    )

    # 11) Save metrics to JSON
    metrics = {
        "recruitment": {
            "auc": auc_r,
            "f1": f1_r,
            "accuracy": acc_r,
            "fairness": {"ΔP": dp_r, "ΔTPR": eo_r},
        },
        "outcome": {
            "auc": auc_o,
            "f1": f1_o,
            "accuracy": acc_o,
            "fairness": {"ΔP": dp_o, "ΔTPR": eo_o},
        },
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 12) ROC curve
    fpr_r, tpr_r, _ = roc_curve(y_te_rec, y_pred_rec)
    fpr_o, tpr_o, _ = roc_curve(y_te_out, y_pred_out)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_r, tpr_r, label=f"Recruit (AUC={auc_r:.2f})")
    plt.plot(fpr_o, tpr_o, label=f"Outcome (AUC={auc_o:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "roc_curve.png"))
    plt.close()

    # 13) SHAP summary for FairOutcomeNet
    explainer = shap.TreeExplainer(fair_model)
    sample = df2_te.sample(n=args.shap_samples, random_state=args.seed)
    shap_vals = explainer.shap_values(Pool(data=sample, cat_features=cat_idx))

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "shap_summary.png"))
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_raw", default="data/raw")
    p.add_argument("--data_processed", default="data/processed")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--shap_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
