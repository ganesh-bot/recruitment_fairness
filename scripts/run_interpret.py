#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from recruitment_fairness.eval.interpret import (
    highlight_phrases,
    summary_plot,
    waterfall_plot,
)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Waterfall subcommand
    w = sub.add_parser("waterfall")
    w.add_argument("--model", required=True, help="Path to model file (.pt or pickle)")
    w.add_argument("--X", required=True, help="Path to X_test.npy")
    w.add_argument("--features", required=True, help="Path to feature_names.json")
    w.add_argument("--idx", type=int, default=0, help="Index of instance to explain")
    w.add_argument("--out", default=None, help="Path to save waterfall plot PNG")

    # Highlight subcommand
    h = sub.add_parser("highlight")
    h.add_argument(
        "--texts", required=True, help="Path to newline-delimited UTF-8 text file"
    )
    h.add_argument("--svs", required=True, help="Path to SHAP values .npy file")
    h.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="HuggingFace tokenizer name or local dir",
    )
    h.add_argument("--top_k", type=int, default=5, help="Number of tokens to highlight")
    h.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of context words around highlighted term",
    )
    h.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of examples to sample; default is all",
    )
    h.add_argument(
        "--out",
        help="Path to write highlight snippets to (UTF-8).If omitted, printto stdout",
    )

    # Summary subcommand
    s = sub.add_parser("summary")
    s.add_argument("--X", required=True, help="Path to X_test.npy")
    s.add_argument("--features", required=True, help="Path to feature_names.json")
    s.add_argument("--svs", help="Optional: precomputed SHAP .npy file")
    s.add_argument("--model", help="Optional: model checkpoint for recompute")
    s.add_argument("--out", help="Path to save summary plot PNG")
    s.add_argument(
        "--max_display", type=int, default=20, help="How many features to show"
    )

    args = parser.parse_args()

    if args.cmd == "waterfall":
        # Load model
        if args.model.endswith(".pt"):
            loaded = torch.load(args.model, map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded:
                from recruitment_fairness.models.fair_outcome_net import (
                    FairOutcomeAdvNet,
                    FairOutcomeNet,
                )

                ckpt = loaded
                net_cls = FairOutcomeAdvNet if "n_groups" in ckpt else FairOutcomeNet
                model = net_cls(
                    input_dim=ckpt.get("input_dim"),
                    hidden_dim=ckpt.get("hidden_dim", 128),
                    dropout=ckpt.get("dropout", 0.2),
                    **(
                        {}
                        if net_cls is FairOutcomeNet
                        else {
                            "n_groups": ckpt.get("n_groups"),
                            "lambda_adv": ckpt.get("lambda_adv", 0.1),
                        }
                    ),
                )
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
            else:
                model = loaded
        else:
            with open(args.model, "rb") as f:
                model = pickle.load(f)

        X = np.load(args.X, allow_pickle=True)
        with open(args.features, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        waterfall_plot(model, X, args.idx, feature_names, out_path=args.out)
        if args.out:
            print(f"✅ Waterfall plot saved to {args.out}")

    elif args.cmd == "highlight":
        # Read texts with UTF-8
        with open(args.texts, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f]
        shap_vals = np.load(args.svs, allow_pickle=True)

        # Optionally sample a subset
        if args.sample_size is not None and args.sample_size < len(texts):
            idxs = random.sample(range(len(texts)), args.sample_size)
            texts = [texts[i] for i in idxs]
            shap_vals = shap_vals[idxs]

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        snippets = highlight_phrases(
            texts,
            shap_vals,
            tokenizer,
            top_k=args.top_k,
            window=args.window,
        )

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as outf:
                for snip in snippets:
                    outf.write(snip + "\n\n")
            print(f"✅ Highlight snippets written to {args.out}")
        else:
            for snip in snippets:
                print(snip)
                print()

    elif args.cmd == "summary":
        X = np.load(args.X, allow_pickle=True)
        with open(args.features, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        if args.svs:
            shap_vals = np.load(args.svs, allow_pickle=True)
            predict_fn = None
        else:
            # load model for recompute
            if args.model.endswith(".pt"):
                loaded = torch.load(args.model, map_location="cpu")
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    from recruitment_fairness.models.fair_outcome_net import (
                        FairOutcomeAdvNet,
                        FairOutcomeNet,
                    )

                    ckpt = loaded
                    net_cls = (
                        FairOutcomeAdvNet if "n_groups" in ckpt else FairOutcomeNet
                    )
                    model = net_cls(
                        input_dim=ckpt.get("input_dim"),
                        hidden_dim=ckpt.get("hidden_dim", 128),
                        dropout=ckpt.get("dropout", 0.2),
                        **(
                            {}
                            if net_cls is FairOutcomeNet
                            else {
                                "n_groups": ckpt.get("n.groups"),
                                "lambda_adv": ckpt.get("lambda_adv", 0.1),
                            }
                        ),
                    )
                    model.load_state_dict(ckpt["state.dict"])
                    model.eval()
                else:
                    model = loaded
            else:
                with open(args.model, "rb") as f:
                    model = pickle.load(f)

            def predict_fn(data):
                arr = np.asarray(data, dtype=np.float32)
                tensor = torch.from_numpy(arr)
                with torch.no_grad():
                    outputs = model(tensor)
                if isinstance(outputs, torch.Tensor):
                    out = outputs
                elif isinstance(outputs, (list, tuple)):
                    out = outputs[0]
                else:
                    return np.array(outputs)
                if out.dim() == 2 and out.size(1) == 2:
                    probs = torch.softmax(out, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(out).squeeze()
                return probs.cpu().numpy()

            shap_vals = None

        summary_plot(
            X,
            feature_names,
            shap_vals,
            max_display=args.max_display,
            out_path=args.out,
        )
        if args.out:
            print(f"✅ Summary plot saved to {args.out}")


if __name__ == "__main__":
    main()
