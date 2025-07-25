# src/recruitment_fairness/data/clinicalbert_embedder.py

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class ClinicalBERTEmbedder:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts, batch_size=16, max_length=128):
        all_vecs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding
                cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_vecs.append(cls_vecs)
        return np.vstack(all_vecs)
