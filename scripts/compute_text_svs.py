#!/usr/bin/env python
import os

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1) Load your pretrained (or fine-tuned) checkpoint
#    If you haven’t fine-tuned a head,
#  this will warn you—and SHAP on a random head isn’t very meaningful.
# MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_ID = "models/text_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,  # adjust if you have more labels
)
model.eval()

# 2) Read in your newline-delimited texts
with open("data/test_texts.txt", encoding="utf-8") as f:
    texts = [line.strip() for line in f]


# 3) Define a predict function for SHAP
def predict_fn(texts_batch):
    # Coerce whatever SHAP hands us into a List[str]
    if isinstance(texts_batch, str):
        texts_batch = [texts_batch]
    elif isinstance(texts_batch, np.ndarray):
        texts_batch = texts_batch.tolist()
    # now safe to tokenize
    # FORCE truncation at BERT’s max length to avoid embedding errors:
    enc = tokenizer(
        texts_batch,
        padding=True,
        truncation=True,
        max_length=512,  # typically 512
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**enc)
    # logits → softmax → P(class=1)
    probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    return probs


# 4) Build the SHAP Text masker + explainer
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_fn, masker)

# 5) Explain all texts (you can lower max_evals for speed)
shap_exp = explainer(texts, max_evals=50)

# 6) Save the raw SHAP values array
os.makedirs("data", exist_ok=True)
np.save("data/test_text_svs.npy", shap_exp.values)
print(f"✅ Wrote SHAP array of shape {shap_exp.values.shape} to data/test_text_svs.npy")
