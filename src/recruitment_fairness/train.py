# src/recruitment_fairness/train.py

import argparse
import os

import numpy as np
import pandas as pd
import torch
from catboost import Pool
from sklearn.decomposition import PCA

from recruitment_fairness.data.clinicalbert_embedder import ClinicalBERTEmbedder
from recruitment_fairness.data.loader import ClinicalTrialsWebCollector
from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor
from recruitment_fairness.models.catboost_net import CatBoostNet
from recruitment_fairness.models.fair_outcome_net import (  # NEW
    train_fair_outcome_net,
    train_fair_outcome_net_adv,
)


def main(args):
    # -- 1) Fetch or load raw data
    collector = ClinicalTrialsWebCollector(output_dir=args.data_raw)
    df_raw = collector.search_trials("", args.max_studies)

    # -- 2) Preprocess & split
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw,
        processed_dir=args.data_processed,
        random_state=args.seed,
    )
    train, val, test = preproc.preprocess(df_raw)

    # # -- Optional subsample for quicker prototyping
    # if len(train) > 1000:
    #     train = train.sample(1000, random_state=args.seed)
    # if len(val) > 200:
    #     val = val.sample(200, random_state=args.seed)

    # -- 3) Structured features
    X_tr_struct, cat_idx = preproc.get_structured_features(train)
    X_va_struct, _ = preproc.get_structured_features(val)

    # -- 4) Text embeddings
    embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
    texts_tr = train["combined_text"].fillna("").astype(str).tolist()
    texts_va = val["combined_text"].fillna("").astype(str).tolist()

    X_tr_text = embedder.embed_texts(
        texts_tr, batch_size=args.batch_size, max_length=args.max_length
    )
    X_va_text = embedder.embed_texts(
        texts_va, batch_size=args.batch_size, max_length=args.max_length
    )

    # -- 5) PCA compression of BERT vectors
    pca = PCA(n_components=50, random_state=args.seed)
    X_tr_txt = pca.fit_transform(X_tr_text)
    X_va_txt = pca.transform(X_va_text)

    # -- 6) Assemble DataFrames & align
    text_cols = [f"text_{i}" for i in range(X_tr_txt.shape[1])]
    df_tr_text = pd.DataFrame(X_tr_txt, index=X_tr_struct.index, columns=text_cols)
    df_va_text = pd.DataFrame(X_va_txt, index=X_va_struct.index, columns=text_cols)

    df_tr = pd.concat([X_tr_struct, df_tr_text], axis=1)
    df_va = pd.concat([X_va_struct, df_va_text], axis=1)
    df_va = df_va.reindex(columns=df_tr.columns, fill_value=0)

    # -- 7) Train RecruitmentNet
    y_tr_rec = train["y_recruit"].to_numpy()
    y_va_rec = val["y_recruit"].to_numpy()

    pool_tr = Pool(data=df_tr, label=y_tr_rec, cat_features=cat_idx)
    pool_va = Pool(data=df_va, label=y_va_rec, cat_features=cat_idx)

    rec_net = CatBoostNet(
        cat_features=cat_idx,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        random_state=args.seed,
    )
    rec_net.fit(
        pool_tr,
        eval_set=pool_va,
        early_stopping_rounds=args.early_stop,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    rec_path = os.path.join(args.model_dir, "recruitment.cbm")
    rec_net.model.save_model(rec_path)
    print(f"✅ Saved RecruitmentNet to {rec_path}")

    # -- 8) Train FairOutcomeNet (second stage)
    rec_score_tr = rec_net.model.predict_proba(pool_tr)[:, 1].reshape(-1, 1)
    rec_score_va = rec_net.model.predict_proba(pool_va)[:, 1].reshape(-1, 1)

    df2_tr = pd.concat(
        [
            df_tr,
            pd.DataFrame(rec_score_tr, index=df_tr.index, columns=["recruit_score"]),
        ],
        axis=1,
    )
    df2_va = pd.concat(
        [
            df_va,
            pd.DataFrame(rec_score_va, index=df_va.index, columns=["recruit_score"]),
        ],
        axis=1,
    )
    df2_va = df2_va.reindex(columns=df2_tr.columns, fill_value=0)

    # -------------------------
    y_tr_out = train["y_outcome"].to_numpy()
    y_va_out = val["y_outcome"].to_numpy()

    if args.model_type == "catboost":
        pool2_tr = Pool(data=df2_tr, label=y_tr_out, cat_features=cat_idx)
        pool2_va = Pool(data=df2_va, label=y_va_out, cat_features=cat_idx)

        out_net = CatBoostNet(
            cat_features=cat_idx,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.lr,
            random_state=args.seed,
        )
        out_net.fit(
            pool2_tr,
            eval_set=pool2_va,
            early_stopping_rounds=args.early_stop,
        )

        out_path = os.path.join(args.model_dir, "fair_outcome.cbm")
        out_net.model.save_model(out_path)
        print(f"✅ Saved FairOutcomeNet(CatBoost) to {out_path}")

    else:
        # --- For MLP: convert string columns to integer categories ---
        for col in ["sponsor_class", "region_income_group", "therapeutic_area"]:
            if col in df2_tr.columns:
                cat_type = pd.Categorical(df2_tr[col])
                df2_tr[col] = cat_type.codes
                df2_va[col] = pd.Categorical(
                    df2_va[col], categories=cat_type.categories
                ).codes

        # ----- MLP branch (with optional adversarial debiasing) -----
        X_train = df2_tr.to_numpy().astype(np.float32)
        X_val = df2_va.to_numpy().astype(np.float32)

        if args.adv_debiasing:
            # convert sensitive attribute to numeric class ids
            g_tr = train[args.group_column].astype("category").cat.codes.to_numpy()
            g_va = val[args.group_column].astype("category").cat.codes.to_numpy()
            n_groups = int(max(g_tr.max(), g_va.max()) + 1)

            fair_net = train_fair_outcome_net_adv(
                X_train=X_train,
                y_train=y_tr_out,
                g_train=g_tr,
                X_val=X_val,
                y_val=y_va_out,
                g_val=g_va,
                input_dim=X_train.shape[1],
                n_groups=n_groups,
                n_epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lambda_adv=args.lambda_adv,
            )
            out_path = os.path.join(args.model_dir, "fair_outcome_adv_mlp.pt")
            torch.save(
                {
                    "state_dict": fair_net.state_dict(),
                    "input_dim": X_train.shape[1],
                    "n_groups": n_groups,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "lambda_adv": args.lambda_adv,
                    "group_column": args.group_column,
                },
                out_path,
            )
            print(f"✅ Saved FairOutcomeNet(MLP+ADV) to {out_path}")
        else:
            fair_net = train_fair_outcome_net(
                X_train=X_train,
                y_train=y_tr_out,
                X_val=X_val,
                y_val=y_va_out,
                input_dim=X_train.shape[1],
                n_epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )
            out_path = os.path.join(args.model_dir, "fair_outcome_mlp.pt")
            torch.save(
                {
                    "state_dict": fair_net.state_dict(),
                    "input_dim": X_train.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
                out_path,
            )
            print(f"✅ Saved FairOutcomeNet(MLP) to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_raw", default="data/raw")
    p.add_argument("--data_processed", default="data/processed")
    p.add_argument("--model_dir", default="models")
    p.add_argument(
        "--model_type",
        choices=["catboost", "mlp"],
        default="catboost",
        help="Final FairOutcomeNet type",
    )
    p.add_argument(
        "--adv_debiasing",
        action="store_true",
        help="Turn on adversarial head for FairOutcomeNet (only --model_type mlp)",
    )
    p.add_argument(
        "--lambda_adv",
        type=float,
        default=0.1,
        help="Strength of adversarial loss (used inside GRL)",
    )
    p.add_argument(
        "--epochs", type=int, default=30, help="Epochs for the MLP FairOutcomeNet"
    )
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--group_column",
        type=str,
        default="sponsor_class",
        help="Sensitive attribute used for adversarial debiasing",
    )
    p.add_argument("--max_studies", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bert_model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HF model for ClinicalBERT embeddings",
    )
    args = p.parse_args()
    main(args)
