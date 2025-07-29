import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
from typing import List, Optional, Callable
from transformers import PreTrainedTokenizer


def waterfall_plot(
    model,
    X: np.ndarray,
    idx: int,
    feature_names: List[str],
    out_path: Optional[str] = None,
    max_display: int = 15,
):
    """
    Generate and optionally save a SHAP waterfall plot for a single instance.
    """
    try:
        X_numeric = np.asarray(X, dtype=float)
    except Exception as e:
        raise ValueError("Input X must be convertible to float.") from e
    masker = shap.maskers.Independent(X_numeric)
    if isinstance(model, torch.nn.Module):
        model.eval()
        def predict_fn(data):
            arr = np.asarray(data, dtype=np.float32)
            tensor = torch.from_numpy(arr)
            with torch.no_grad(): outputs = model(tensor)
            if isinstance(outputs, torch.Tensor): out = outputs
            elif isinstance(outputs, (list, tuple)): out = outputs[0]
            else: return np.array(outputs)
            if out.dim() == 2 and out.size(1) == 2: probs = torch.softmax(out, dim=1)[:,1]
            else: probs = torch.sigmoid(out).squeeze()
            return probs.cpu().numpy()
    elif hasattr(model,"predict_proba"): predict_fn = lambda d: model.predict_proba(d)[:,1]
    elif hasattr(model,"predict"): predict_fn = model.predict
    else: raise ValueError("Model must implement predict/prob or be nn.Module.")
    expl = shap.Explainer(predict_fn, masker, feature_names=feature_names)
    shap_values = expl(X_numeric)
    instance_sv = shap_values[idx]
    shap.plots.waterfall(instance_sv, max_display=max_display, show=False)
    fig = plt.gcf()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _merge_subwords(tokens: List[str], values: np.ndarray):
    words, word_vals = [], []
    buf_tok, buf_val = tokens[0], values[0]
    for tok, val in zip(tokens[1:], values[1:]):
        if tok.startswith("##"): buf_tok += tok[2:]; buf_val += val
        else: words.append(buf_tok); word_vals.append(buf_val); buf_tok, buf_val = tok, val
    words.append(buf_tok); word_vals.append(buf_val)
    return words, np.array(word_vals)


def highlight_phrases(
    texts: List[str], shap_values: np.ndarray,
    tokenizer: PreTrainedTokenizer, top_k: int = 5, window: int = 5,
) -> List[str]:
    highlighted_texts = []
    for text, sv in zip(texts, shap_values):
        subtoks = tokenizer.tokenize(text)
        if len(subtoks) != len(sv):
            min_len = min(len(subtoks), len(sv)); subtoks, sv = subtoks[:min_len], sv[:min_len]
        words, vals = _merge_subwords(subtoks, sv)
        top_inds = np.argsort(np.abs(vals))[-top_k:]
        snippets = []
        for idx in sorted(top_inds):
            start = max(0, idx-window); end = min(len(words), idx+window+1)
            parts = []
            for i in range(start, end):
                w = words[i]
                parts.append(f"**{w}**" if i==idx else w)
            snippets.append("..." + " ".join(parts) + "...")
        highlighted_texts.append("\n".join(snippets))
    return highlighted_texts


def summary_plot(
    X: np.ndarray,
    feature_names: List[str],
    shap_values: np.ndarray,
    max_display: int = 20,
    out_path: Optional[str] = None,
):
    """
    Bar plot of the top max_display features by mean absolute SHAP value.

    :param X: numpy array of shape (n_samples, n_features)
    :param feature_names: list of feature names length n_features
    :param shap_values: numpy array of SHAP values shape (n_samples, n_features)
    :param max_display: how many top features to display
    :param out_path: if provided, path to write the PNG
    """
    import matplotlib.pyplot as plt

    # Unwrap if shap_values is a list (binary classifier)
    if isinstance(shap_values, (list, tuple)) and len(shap_values) == 2:
        vals = shap_values[1]
    else:
        vals = np.asarray(shap_values)

    # Squeeze last singleton dimension if present
    vals = np.squeeze(vals)
    # Ensure 2D
    if vals.ndim != 2:
        raise ValueError(f"Expected 2D shap_values, got shape {vals.shape}")
        raise ValueError(f"Expected 2D shap_values, got shape {vals.shape}")

    # Compute mean absolute importance
    imp = np.mean(np.abs(vals), axis=0)

    # Select top features
    idx = np.argsort(imp)[::-1][:max_display]
    top_imp = imp[idx]
    top_feats = [feature_names[i] for i in idx]

    # Plot horizontal bar chart
    plt.figure(figsize=(8, max_display * 0.3 + 1))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_imp)
    plt.yticks(y_pos, top_feats)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance")
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
