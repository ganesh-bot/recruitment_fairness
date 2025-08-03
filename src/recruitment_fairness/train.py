#!/usr/bin/env python
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import Pool
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from recruitment_fairness.data.clinicalbert_embedder import ClinicalBERTEmbedder
from recruitment_fairness.data.loader import ClinicalTrialsWebCollector
from recruitment_fairness.data.preprocess import ClinicalTrialPreprocessor
from recruitment_fairness.models.catboost_net import CatBoostNet
from recruitment_fairness.models.fair_outcome_net import (
    train_fair_outcome_net,
    train_fair_outcome_net_adv,
)

# 0) Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # 1) Fetch or load raw data
    raw_dir = Path(args.data_raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_collect or list(raw_dir.glob("raw_clinical_trials_*.csv")):
        # reuse the latest existing CSV
        preproc_helper = ClinicalTrialPreprocessor(
            data_dir=args.data_raw, processed_dir=args.data_processed, random_state=args.seed
        )
        df_raw = preproc_helper.load_latest_raw()
    else:
        # first time: fetch from API
        collector = ClinicalTrialsWebCollector(output_dir=args.data_raw)
        df_raw = collector.search_trials("", args.max_studies)

    # 2) Preprocess & split (80/10/10)
    preproc = ClinicalTrialPreprocessor(
        data_dir=args.data_raw,
        processed_dir=args.data_processed,
        random_state=args.seed,
    )
    train, val, test = preproc.preprocess(df_raw)

    # 3) Extract structured features + cat indices
    X_tr_struct, cat_idx = preproc.get_structured_features(train)
    X_va_struct, _       = preproc.get_structured_features(val)

    # 4) Optionally drop the recruitment/text branch
    if args.drop_recruitment:
        print("⚠️  Dropping recruitment/text features per --drop_recruitment")

    # 5) Text embeddings + PCA (skip if drop_recruitment)
    if not args.drop_recruitment:
        embedder = ClinicalBERTEmbedder(model_name=args.bert_model)
        embedder.model.to(device)
        texts_tr = train["combined_text"].fillna("").astype(str).tolist()
        texts_va = val["combined_text"].fillna("").astype(str).tolist()

        X_tr_text = embedder.embed_texts(
            texts_tr, batch_size=args.batch_size, max_length=args.max_length
        )
        X_va_text = embedder.embed_texts(
            texts_va, batch_size=args.batch_size, max_length=args.max_length
        )

        pca = PCA(n_components=50, random_state=args.seed)
        X_tr_txt = pca.fit_transform(X_tr_text)
        X_va_txt = pca.transform(X_va_text)

    # 6) Assemble DataFrames & align
    if args.drop_recruitment:
        df_tr = X_tr_struct.copy()
        df_va = X_va_struct.copy()
    else:
        text_cols  = [f"text_{i}" for i in range(X_tr_txt.shape[1])]
        df_tr_text = pd.DataFrame(X_tr_txt, index=X_tr_struct.index, columns=text_cols)
        df_va_text = pd.DataFrame(X_va_txt, index=X_va_struct.index, columns=text_cols)

        df_tr = pd.concat([X_tr_struct, df_tr_text], axis=1)
        df_va = pd.concat([X_va_struct, df_va_text], axis=1)
        df_va = df_va.reindex(columns=df_tr.columns, fill_value=0)

    # 7) RecruitmentNet or structured-only baseline
    if not args.drop_recruitment:
        # 7a) Train RecruitmentNet
        y_tr_rec = train["y_recruit"].to_numpy()
        y_va_rec = val["y_recruit"].to_numpy()
        pool_tr  = Pool(data=df_tr, label=y_tr_rec, cat_features=cat_idx)
        pool_va  = Pool(data=df_va, label=y_va_rec, cat_features=cat_idx)

        rec_net = CatBoostNet(
            cat_features=cat_idx,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.lr,
            random_state=args.seed,
            auto_class_weights='Balanced'
        )
        rec_net.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=args.early_stop)

        os.makedirs(args.model_dir, exist_ok=True)
        rec_path = os.path.join(args.model_dir, "recruitment.cbm")
        os.makedirs(args.model_dir, exist_ok=True)
        rec_net.model.save_model(rec_path)
        print(f"✅ Saved RecruitmentNet to {rec_path}")

        # 7b) Prepare second-stage inputs
        rec_score_tr = rec_net.model.predict_proba(pool_tr)[:, 1].reshape(-1, 1)
        rec_score_va = rec_net.model.predict_proba(pool_va)[:, 1].reshape(-1, 1)
        df2_tr       = pd.concat(
            [df_tr, pd.DataFrame(rec_score_tr, index=df_tr.index, columns=["recruit_score"])],
            axis=1,
        )
        df2_va       = pd.concat(
            [df_va, pd.DataFrame(rec_score_va, index=df_va.index, columns=["recruit_score"])],
            axis=1,
        )
        df2_va = df2_va.reindex(columns=df2_tr.columns, fill_value=0)
    else:
        # Baseline: structured-only
        print("⚠️  Skipping RecruitmentNet; using structured-only inputs for outcome.")
        df2_tr = df_tr.copy()
        df2_va = df_va.copy()

    # 8) Outcome model training
    y_tr_out = train["y_outcome"].to_numpy()
    y_va_out = val["y_outcome"].to_numpy()

    if args.model_type == "catboost":
        pool2_tr = Pool(data=df2_tr, label=y_tr_out, cat_features=cat_idx)
        pool2_va = Pool(data=df2_va, label=y_va_out, cat_features=cat_idx)
        out_net  = CatBoostNet(
            cat_features=cat_idx,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.lr,
            random_state=args.seed,
        )
        out_net.fit(pool2_tr, eval_set=pool2_va, early_stopping_rounds=args.early_stop)

        out_path = os.path.join(args.model_dir, "fair_outcome.cbm")
        os.makedirs(args.model_dir, exist_ok=True)
        out_net.model.save_model(out_path)
        print(f"✅ Saved FairOutcomeNet(CatBoost) to {out_path}")
    else:
        # 8b) MLP branch (with optional adversarial debiasing)
        # convert categorical columns to integer codes
        for col in ["sponsor_class", "region_income_group", "therapeutic_area"]:
            if col in df2_tr.columns:
                cat_type     = pd.Categorical(df2_tr[col])
                df2_tr[col]  = cat_type.codes
                df2_va[col]  = pd.Categorical(df2_va[col], categories=cat_type.categories).codes

        X_train = torch.from_numpy(df2_tr.to_numpy().astype(np.float32)).to(device)
        X_val   = torch.from_numpy(df2_va.to_numpy().astype(np.float32)).to(device)

        if args.adv_debiasing:
            g_tr     = torch.from_numpy(train[args.group_column].astype("category").cat.codes.to_numpy()).to(device)
            g_va     = torch.from_numpy(val[args.group_column].astype("category").cat.codes.to_numpy()).to(device)
            n_groups = int(max(g_tr.max(), g_va.max()) + 1)
            fair_net = train_fair_outcome_net_adv(
                X_train=X_train, y_train=y_tr_out,
                g_train=g_tr, X_val=X_val, y_val=y_va_out, g_val=g_va,
                input_dim=X_train.shape[1], n_groups=n_groups,
                n_epochs=args.epochs, lr=args.lr,
                hidden_dim=args.hidden_dim, dropout=args.dropout,
                lambda_adv=args.lambda_adv,
            )
            out_path = os.path.join(args.model_dir, "fair_outcome_adv_mlp.pt")
            torch.save({
                "state_dict": fair_net.state_dict(),
                "input_dim":  X_train.shape[1],
                "n_groups":   n_groups,
                "hidden_dim": args.hidden_dim,
                "dropout":    args.dropout,
                "lambda_adv": args.lambda_adv,
                "group_column": args.group_column,
            }, out_path)
            print(f"✅ Saved FairOutcomeNet(MLP+ADV) to {out_path}")
        else:
            fair_net = train_fair_outcome_net(
                X_train=X_train, y_train=y_tr_out,
                X_val=X_val, y_val=y_va_out,
                input_dim=X_train.shape[1], n_epochs=args.epochs,
                lr=args.lr, hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(device)
            out_path = os.path.join(args.model_dir, "fair_outcome_mlp.pt")
            torch.save({
                "state_dict": fair_net.state_dict(),
                "input_dim":  X_train.shape[1],
                "hidden_dim": args.hidden_dim,
                "dropout":    args.dropout,
            }, out_path)
            print(f"✅ Saved FairOutcomeNet(MLP) to {out_path}")

    # 9) Optional text-classifier fine-tuning (skip if drop_recruitment)
    if args.train_text_model and not args.drop_recruitment:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        # 9a) Dataset helper
        class TextDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels    = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {}
                for k, v in self.encodings.items():
                    item[k] = v[idx] if isinstance(v, torch.Tensor) else torch.tensor(v[idx])
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        # 9b) Load & freeze BERT
        tokenizer  = AutoTokenizer.from_pretrained(args.bert_model)
        text_model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=2).to(device)
        for name, param in text_model.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False
        text_model.train()

        # 9c) Tokenize + Trainer setup
        train_texts  = train["combined_text"].fillna("").astype(str).tolist()
        val_texts    = val["combined_text"].fillna("").astype(str).tolist()
        train_labels = train["y_outcome"].fillna(0).astype(int).tolist()
        val_labels   = val["y_outcome"].fillna(0).astype(int).tolist()

        train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        val_enc   = tokenizer(val_texts,   padding=True, truncation=True, max_length=256, return_tensors="pt")

        train_ds = TextDataset(train_enc, train_labels)
        val_ds   = TextDataset(val_enc,   val_labels)

        training_args = TrainingArguments(
            output_dir=args.text_output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            fp16=True,
            do_train=True,
            do_eval=True,
            save_strategy="epoch",
            dataloader_pin_memory=False,
        )
        trainer = Trainer(model=text_model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        os.makedirs(args.text_output_dir, exist_ok=True)
        trainer.save_model(args.text_output_dir)
        tokenizer.save_pretrained(args.text_output_dir)
        print(f"✅ Saved text classifier to {args.text_output_dir}")

    # 10) Dump test‐set for SHAP
    X_te_struct, _ = preproc.get_structured_features(test)
    if not args.drop_recruitment:
        texts_te = test["combined_text"].fillna("").astype(str).tolist()
        X_te_text = embedder.embed_texts(texts_te, batch_size=args.batch_size, max_length=args.max_length)
        X_te_txt  = pca.transform(X_te_text)
        text_cols = [f"text_{i}" for i in range(X_te_txt.shape[1])]
        df_te_text = pd.DataFrame(X_te_txt, index=X_te_struct.index, columns=text_cols)
        df_te      = pd.concat([X_te_struct, df_te_text], axis=1)
    else:
        df_te = X_te_struct.copy()
    # align columns
    df_te = df_te.reindex(columns=df_tr.columns, fill_value=0)

    os.makedirs(args.model_dir, exist_ok=True)
    np.save(os.path.join(args.model_dir, "X_test.npy"), df_te.to_numpy())
    with open(os.path.join(args.model_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(df_tr.columns.tolist(), f, indent=2)
    print(f"✅ Wrote X_test.npy and feature_names.json to {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop_recruitment",
        action="store_true",
        help="Skip text/recruitment features and train on structured inputs only",
    )
    parser.add_argument("--data_raw", default="data/raw")
    parser.add_argument("--data_processed", default="data/processed")
    parser.add_argument(
        "--skip_collect",
        action="store_true",
        help="If set, do not fetch new trials; load the latest CSV from --data_raw"
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument(
        "--model_type",
        choices=["catboost", "mlp"],
        default="catboost",
        help="Final outcome model type",
    )
    parser.add_argument(
        "--adv_debiasing",
        action="store_true",
        help="Use adversarial debiasing head (MLP only)",
    )
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--train_text_model",
        action="store_true",
        help="Also fine-tune ClinicalBERT for text classification",
    )
    parser.add_argument(
        "--text_output_dir",
        default="models/text_classifier",
        help="Save directory for text classifier",
    )
    parser.add_argument("--group_column", type=str, default="sponsor_class")
    parser.add_argument("--max_studies", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bert_model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model for ClinicalBERT embeddings",
    )
    args = parser.parse_args()
    main(args)
