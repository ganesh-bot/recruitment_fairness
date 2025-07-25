# src/recruitment_fairness/train.py

import argparse
import os

# import numpy as np
import pandas as pd
from catboost import Pool

from recruitment_fairness.data.clinicalbert_embedder import ClinicalBERTEmbedder
from recruitment_fairness.data.loader import ClinicalTrialsWebCollector
from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor
from recruitment_fairness.models.catboost_net import CatBoostNet


def main(args):
    # 1. Collect (or load cached)
    collector = ClinicalTrialsWebCollector(output_dir=args.data_raw)
    df = collector.search_trials("", args.max_studies)

    # 2. Preprocess & split
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw, processed_dir=args.data_processed
    )
    train, val, test = preproc.preprocess(df)

    # 3. Structured features
    X_tr_struct, cat_idx = preproc.get_structured_features(train)
    X_va_struct, _ = preproc.get_structured_features(val)

    # 4. Text embeddings
    embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
    X_tr_text = embedder.embed_texts(
        train["brief_summary"].fillna("").tolist(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    X_va_text = embedder.embed_texts(
        val["brief_summary"].fillna("").tolist(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 5. Assemble DataFrames & Pools for CatBoost, aligning val/test to train columns
    # 5.1 Build train DataFrame
    df_text_tr = pd.DataFrame(
        X_tr_text,
        columns=[f"text_{i}" for i in range(X_tr_text.shape[1])],
        index=X_tr_struct.index,
    )
    X_tr_df = pd.concat([X_tr_struct, df_text_tr], axis=1)

    # 5.2 Build val DataFrame only (we don't need test here)
    df_text_va = pd.DataFrame(
        X_va_text,
        columns=[f"text_{i}" for i in range(X_va_text.shape[1])],
        index=X_va_struct.index,
    )
    X_va_df = pd.concat([X_va_struct, df_text_va], axis=1)

    # 5.3 Align val to train columns (fill missing dummies with 0)
    X_va_df = X_va_df.reindex(columns=X_tr_df.columns, fill_value=0)

    # 5.4 Build Pools
    train_pool = Pool(
        data=X_tr_df,
        label=train["is_success"].to_numpy(),
        cat_features=["sponsor_class"],
    )
    val_pool = Pool(
        data=X_va_df, label=val["is_success"].to_numpy(), cat_features=["sponsor_class"]
    )

    # 6. Train RecruitmentNet
    recruit_model = CatBoostNet(
        cat_features=["sponsor_class"],
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        random_state=args.seed,
    )
    recruit_model.fit(
        train_pool, eval_set=val_pool, early_stopping_rounds=args.early_stop
    )
    recruit_model.model.save_model(os.path.join(args.model_dir, "recruitment.cbm"))

    # 7. Second‐stage FairOutcomeNet
    # Extend features with recruitment score
    X_fa_train = recruit_model.model.predict_proba(train_pool)[:, 1].reshape(-1, 1)
    X_fa_val = recruit_model.model.predict_proba(val_pool)[:, 1].reshape(-1, 1)
    # build new Pools
    # build DataFrames for second‐stage
    df2_tr = pd.concat(
        [
            X_tr_df,
            pd.DataFrame(X_fa_train, columns=["recruit_score"], index=X_tr_df.index),
        ],
        axis=1,
    )
    df2_va = pd.concat(
        [
            X_va_df,
            pd.DataFrame(X_fa_val, columns=["recruit_score"], index=X_va_df.index),
        ],
        axis=1,
    )
    # align to training DF columns
    df2_va = df2_va.reindex(columns=df2_tr.columns, fill_value=0)

    train_pool2 = Pool(
        data=df2_tr,
        label=train["is_success"].to_numpy(),
        cat_features=["sponsor_class"],
    )
    val_pool2 = Pool(
        data=df2_va, label=val["is_success"].to_numpy(), cat_features=["sponsor_class"]
    )

    fair_model = CatBoostNet(
        cat_features=["sponsor_class"],
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        random_state=args.seed,
    )
    fair_model.fit(
        train_pool2, eval_set=val_pool2, early_stopping_rounds=args.early_stop
    )
    fair_model.model.save_model(os.path.join(args.model_dir, "fair_outcome.cbm"))

    print("✅ Training complete; models saved to", args.model_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_raw", default="data/raw")
    p.add_argument("--data_processed", default="data/processed")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--max_studies", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT")
    args = p.parse_args()
    main(args)
