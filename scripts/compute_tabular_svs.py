#!/usr/bin/env python
import os,argparse

import numpy as np
import shap
import torch

from recruitment_fairness.models.fair_outcome_net import FairOutcomeNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0) parse args
parser = argparse.ArgumentParser()
parser.add_argument("--X",         required=True,
                    help="Path to numeric X_test.npy")
parser.add_argument("--svs",       required=True,
                    help="Where to write shap_tabular_svs.npy")
parser.add_argument("--background_size", type=int, default=100)
args = parser.parse_args()

# 1) Load your test features
X = np.load(args.X, allow_pickle=True)
try:
    X = X.astype(np.float32)
except Exception as e:
    raise ValueError(f"Your X contains non‐numeric values: {e}")

# 2) Load your MLP checkpoint and rebuild the model if needed
ckpt = torch.load("models/fair_outcome_mlp.pt", map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    base_model = FairOutcomeNet(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt.get("hidden_dim", 128),
        dropout=ckpt.get("dropout", 0.2),
    )
    base_model.load_state_dict(ckpt["state_dict"])
else:
    base_model = ckpt
base_model = base_model.to(device)
base_model.eval()


# 3) Wrap model to output probabilities (single column)
class ProbModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        logits = self.base(x)
        if isinstance(logits, torch.Tensor):
            if logits.dim() == 2 and logits.size(1) == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits).squeeze()
        else:
            return torch.tensor(logits)
        return probs.unsqueeze(-1)


prob_model = ProbModel(base_model)
prob_model = prob_model.to(device)
prob_model.eval()

# 4) Prepare background and input tensors
bg_idxs = np.random.choice(len(X), size=min(100, len(X)), replace=False)
background = torch.from_numpy(X[bg_idxs].astype(np.float32)).to(device)
X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)

# 5) Build DeepExplainer on probability model
explainer = shap.DeepExplainer(prob_model, background)

# 6) Compute SHAP values with additivity check disabled
shap_vals = explainer.shap_values(X_tensor, check_additivity=False)
# shap_vals may be list of arrays or single array
if isinstance(shap_vals, (list, tuple)):
    shap_vals = shap_vals[0]

# 7) Save SHAP values
os.makedirs(os.path.dirname(args.svs), exist_ok=True)
np.save(args.svs, shap_vals)
print(f"✅ Wrote SHAP values of shape {shap_vals.shape} to models/shap_tabular_svs.npy")
