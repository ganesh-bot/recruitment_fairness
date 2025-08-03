#!/usr/bin/env python
"""
evaluate.py: Metrics‐only evaluation (and plotting) for all three baselines.

Usage:

1) Static CatBoost baseline:
   python evaluate.py \
     --data_raw data/raw \
     --data_processed data/processed \
     --model_dir models/static_outcome \
     --model_type catboost \
     --metrics_only

2) Structured‐only MLP baseline:
   python evaluate.py \
     --data_raw data/raw \
     --data_processed data/processed \
     --model_dir models/mlp_structured \
     --model_type mlp \
     --metrics_only

3) Full RecruitmentNet:
   python evaluate.py \
     --data_raw data/raw \
     --data_processed data/processed \
     --model_dir models/recruitment_net \
     --model_type mlp \
     --metrics_only
"""
import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, mean_absolute_error
)
from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor
from recruitment_fairness.data.clinicalbert_embedder import ClinicalBERTEmbedder
from recruitment_fairness.models.fair_outcome_net import FairOutcomeNet, FairOutcomeAdvNet

from recruitment_fairness.eval.plotting import plot_delta_tpr, plot_roc_curves, plot_delta_tpr,fairness_report_by_group

from recruitment_fairness.eval.fairness_audit import audit_groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Load train/test tables
    train = pd.read_csv(os.path.join(args.data_processed, "train.csv"), index_col=0)
    test  = pd.read_csv(os.path.join(args.data_processed, "test.csv"),  index_col=0)

    # 2) Get structured‐only DataFrames
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw,
        processed_dir=args.data_processed,
        random_state=args.seed
    )
    df_tr_struct, cat_idx = preproc.get_structured_features(train)
    df_te_struct, _       = preproc.get_structured_features(test)

    # 3) Decide if this is structured‐only MLP baseline
    structured_only = (
        args.model_type == "mlp"
        and not os.path.exists(os.path.join(args.model_dir, "recruitment.cbm"))
    )

    if structured_only:
        # purely structured features
        df_tr = df_tr_struct.copy()
        df_te = df_te_struct.copy()
        print(f"ℹ️ Structured‐only mode: {df_te.shape[1]} features")
    else:
        # 4) Full model: text embed + PCA
        embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
        texts_tr = train["combined_text"].fillna("").astype(str).tolist()
        texts_te = test["combined_text"].fillna("").astype(str).tolist()
        X_tr_text = embedder.embed_texts(texts_tr, batch_size=args.batch_size, max_length=args.max_length)
        X_te_text = embedder.embed_texts(texts_te, batch_size=args.batch_size, max_length=args.max_length)

        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        X_tr_txt = pca.fit_transform(X_tr_text)
        X_te_txt = pca.transform(X_te_text)

        # 5) Assemble full feature DataFrames
        text_cols = [f"text_{i}" for i in range(X_tr_txt.shape[1])]
        df_tr = pd.concat(
            [df_tr_struct.reset_index(drop=True), pd.DataFrame(X_tr_txt, columns=text_cols)],
            axis=1
        )
        df_te = pd.concat(
            [df_te_struct.reset_index(drop=True), pd.DataFrame(X_te_txt, columns=text_cols)],
            axis=1
        )
        print(f"ℹ️ Full‐model mode: {df_te.shape[1]} features")

    # 6) True labels & groups
    y_te_rec   = test["y_recruit"].to_numpy()
    y_te_out   = test["y_outcome"].to_numpy()
    group_test = test[args.group_column].to_numpy()
    for g in np.unique(group_test):
        mask = group_test == g
        pos = (y_te_out[mask] == 1).sum()
        neg = (y_te_out[mask] == 0).sum()
        print(f"Group {g!r}: {pos} positives, {neg} negatives")

      # 7) Load RecruitmentNet if exists
    rec_path = os.path.join(args.model_dir, "recruitment.cbm")
    rec_model = None
    if os.path.exists(rec_path):
        rec_model = CatBoostClassifier()
        rec_model.load_model(rec_path)

    # 8) Load outcome model
    is_mlp = args.model_type == "mlp"
    if not is_mlp:
        fair_model = CatBoostClassifier()
        fair_model.load_model(os.path.join(args.model_dir, "fair_outcome.cbm"))
    else:
        # pick adv‐mlp if present
        adv_file = os.path.join(args.model_dir, "fair_outcome_adv_mlp.pt")
        std_file = os.path.join(args.model_dir, "fair_outcome_mlp.pt")
        ckpt_file = adv_file if os.path.exists(adv_file) else std_file
        ckpt      = torch.load(ckpt_file, map_location=device)
        if "n_groups" in ckpt:
            fair_model = FairOutcomeAdvNet(
                input_dim=ckpt["input_dim"],
                n_groups=ckpt["n_groups"],
                hidden_dim=ckpt.get("hidden_dim",128),
                dropout=ckpt.get("dropout",0.2),
                lambda_adv=ckpt.get("lambda_adv",0.1),
            )
        else:
            fair_model = FairOutcomeNet(
                input_dim=ckpt["input_dim"],
                hidden_dim=ckpt.get("hidden_dim",128),
                dropout=ckpt.get("dropout",0.2),
            )
        fair_model.load_state_dict(ckpt["state_dict"])
        fair_model.to(device)
        fair_model.eval()

    # 9) Recruitment predictions & metrics
    if rec_model:
        pool_r = Pool(data=df_te, cat_features=cat_idx)
        y_pred_rec = rec_model.predict_proba(pool_r)[:,1]
        auc_rec = roc_auc_score(y_te_rec, y_pred_rec)
        f1_rec  = f1_score(y_te_rec, y_pred_rec>0.5)
        acc_rec  = accuracy_score(y_te_rec, y_pred_rec>0.5)
    else:
        y_pred_rec = None
        auc_rec = f1_rec = acc_rec = None

    # 10) Outcome predictions & metrics
    if not is_mlp:
        df_te2 = df_te.copy()
        if y_pred_rec is not None:
            df_te2["recruit_score"] = y_pred_rec
        pool_o = Pool(data=df_te2, cat_features=cat_idx)
        y_pred_out = fair_model.predict_proba(pool_o)[:,1]
    else:
        df_te2 = df_te.copy()
        if y_pred_rec is not None:
            df_te2["recruit_score"] = y_pred_rec
        # encode any categorical group field
        df_te2[args.group_column] = pd.Categorical(df_te2[args.group_column]).codes
        # --- ENCODE ALL STRING/CATEGORICAL COLUMNS to numeric codes ----
        for col in df_te2.columns:
            if df_te2[col].dtype == "object" or pd.api.types.is_categorical_dtype(df_te2[col]):
                df_te2[col] = pd.Categorical(df_te2[col]).codes
        # ----------------------------------------------------------------
        X_out = torch.tensor(df_te2.to_numpy().astype(np.float32)).to(device)
        with torch.no_grad():
            out = fair_model(X_out)
            # unpack if adversarial returns (main, adv)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            # bring predictions back to CPU for numpy
            y_pred_out = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)


    auc_out = roc_auc_score(y_te_out, y_pred_out)
    f1_out  = f1_score(y_te_out, y_pred_out>0.5)
    acc_out = accuracy_score(y_te_out, y_pred_out>0.5)

    # at the end of main(), before metrics_only return
    np.save(os.path.join(args.model_dir, "y_true_out.npy"), y_te_out)
    np.save(os.path.join(args.model_dir, "y_pred_out.npy"), y_pred_out)


    # Fairness ΔTPR
    # New modular code using fairness_audit.py
    group_labels = {args.group_column: group_test}

    fairness_report = audit_groups(
        y_true=y_te_out,
        y_pred=(y_pred_out > 0.5).astype(int),
        group_labels=group_labels,
        y_proba=y_pred_out  # for AUC computation
    )

    # Print detailed fairness metrics
    for group, metrics in fairness_report['per_group'][args.group_column].items():
        print(f"\nMetrics for Group '{group}':")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")

    # Print Delta Metrics
    print("\n=== Fairness Δ Metrics ===")
    for delta_metric, delta_values in fairness_report.items():
        if delta_metric.startswith('Δ'):
            delta_val = delta_values[args.group_column]
            print(f"{delta_metric}: {delta_val}")

    # --- Save summary JSON even in metrics_only mode ---
    summary = pd.DataFrame([{
        "model": os.path.basename(args.model_dir),
        "model_type": args.model_type,
        "outcome_auc": auc_out,
        "outcome_f1": f1_out,
        "outcome_acc": acc_out,
        "recruit_auc": auc_rec,
        "recruit_f1": f1_rec,
        "recruit_acc": acc_rec,
        "recruit_mae": (mean_absolute_error(y_te_rec, y_pred_rec) if rec_model else None),
        "delta_tpr": fairness_report["ΔTPR"][args.group_column],
        "delta_fpr": fairness_report["ΔFPR"][args.group_column],
        "delta_auc": fairness_report["ΔAUC"][args.group_column],
        "delta_positive_rate": fairness_report["ΔP"][args.group_column],
    }])

    out_json = os.path.join(args.model_dir, "metrics_summary.json")
    summary.to_json(out_json, orient="records", indent=2)
    print(f"✅ Wrote enhanced fairness summary JSON to {out_json}")

    # 11) Metrics-only printout
    if args.metrics_only:
        print("\n=== Metrics Only ===")
        if rec_model:
            print(f"Recruitment → AUC {auc_rec:.3f}, F1 {f1_rec:.3f}, Acc {acc_rec:.3f}")
            print(f"Recruitment MAE: {mean_absolute_error(y_te_rec, y_pred_rec):.3f}")
        else:
            print("Recruitment → skipped")
        print(f"Outcome     → AUC {auc_out:.3f}, F1 {f1_out:.3f}, Acc {acc_out:.3f}")

        return

    # # --- 12) SUMMARY TABLE & JSON FOR REPORT ---
    # # Compute delta_tpr if not already
    # if 'tprs' not in locals():
    #     tprs = []
    #     for g in np.unique(group_test):
    #         mask = group_test == g
    #         tp = ((y_te_out[mask]==1)&(y_pred_out[mask]>0.5)).sum()
    #         fn = ((y_te_out[mask]==1)&(y_pred_out[mask]<=0.5)).sum()
    #         tprs.append(tp/(tp+fn) if tp+fn>0 else np.nan)
    # delta_tpr = float(np.nanmax(tprs) - np.nanmin(tprs))

    # # Build a DataFrame summarizing all three baselines
    # # You can parametrize this depending on which model_dir you're in,
    # # but here’s the general pattern:
    # summary = pd.DataFrame([{
    #     "model":      os.path.basename(args.model_dir),
    #     "model_type": args.model_type,
    #     "outcome_auc":   auc_out,
    #     "outcome_f1":    f1_out,
    #     "outcome_acc":   acc_out,
    #     "recruit_auc":   auc_rec,
    #     "recruit_f1":    f1_rec,
    #     "recruit_acc":   acc_rec,
    #     "recruit_mae":   (mean_absolute_error(y_te_rec, y_pred_rec) if rec_model else None),
    #     "delta_tpr":     delta_tpr,
    # }])

    # # Print it as a table
    # print("\n=== Summary Table ===")
    # print(summary.to_markdown(index=False))

    # # Save to JSON
    # out_json = os.path.join(args.model_dir, "metrics_summary.json")
    # summary.to_json(out_json, orient="records", indent=2)
    # print(f"✅ Wrote summary JSON to {out_json}")
    # # ------------------------------------------------


    # 13) Full plotting logic goes here (unchanged)…
    # after summary creation:
    plot_delta_tpr(summary, save_path=os.path.join(args.model_dir, "delta_tpr.png"))

    # 13a) Plot ΔTPR bar chart
    df_summary = pd.DataFrame([{
        "model": args.model_dir.split("/")[-1],
        "delta_tpr": delta_tpr,
    }])
    plot_delta_tpr(df_summary, save_path=os.path.join(args.model_dir, "deltatpr.png"))

    # 12b) Plot ROC curves
    # --- compute ROC curves for plotting ---
    from sklearn.metrics import roc_curve
    fpr_out, tpr_out, _ = roc_curve(y_te_out, y_pred_out)
    roc_data = {"Outcome": (fpr_out, tpr_out)}

    if rec_model:
        fpr_rec, tpr_rec, _ = roc_curve(y_te_rec, y_pred_rec)
        roc_data["Recruitment"] = (fpr_rec, tpr_rec)
    plot_roc_curves(roc_data, save_path=os.path.join(args.model_dir, "roc_curves.png"))

    # assemble a small DataFrame for fairness
    df_fair = pd.DataFrame({
        "y_true":  y_te_out,
        "y_pred":  (y_pred_out > 0.5).astype(int),
        "sponsor_class":      test["sponsor_class"].astype(str),
        "therapeutic_area":   test["therapeutic_area"].astype(str),
        "region_income_group": test["region_income_group"].astype(str),
    })

    fairness_report_by_group(
        df=df_fair,
        true_col="y_true",
        pred_col="y_pred",
        group_cols=["sponsor_class", "therapeutic_area", "region_income_group"],
        out_dir=os.path.join(args.model_dir, "fairness_reports")
    )    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_raw",       required=True)
    parser.add_argument("--data_processed", required=True)
    parser.add_argument("--model_dir",      required=True)
    parser.add_argument(
        "--model_type",
        choices=["catboost","mlp"],
        required=True
    )
    parser.add_argument(
        "--group_column",
        default="sponsor_class",
        help="Column for fairness grouping"
    )
    parser.add_argument(
        "--bert_model",
        default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pca_dim",    type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Print metrics and exit"
    )
    args = parser.parse_args()
    main(args)
