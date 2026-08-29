"""
extraction/bert_extractor.py
------------------------------
Fine-tunes and evaluates a HuggingFace sequence-classification model as a
*multi-label* classifier over config.PRODUCT_LABELS: given an email body,
predict the set of products it references. This is the BERT side of the
product-extraction comparison (mirrors bert_classifier.py's structure but
uses a sigmoid multi-label head instead of a single-label softmax one,
since an email can legitimately reference more than one product).
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, hamming_loss


def _multi_hot(products, label_list):
    label2idx = {l: i for i, l in enumerate(label_list)}
    vec = [0.0] * len(label_list)
    for p in products:
        vec[label2idx[p]] = 1.0
    return vec


def train_bert_extractor(X_train, y_train, X_test, y_test, label_list,
                          model_name="distilbert-base-uncased",
                          epochs=3, batch_size=16, device=None, threshold=0.5):
    """
    Fine-tunes a pretrained transformer checkpoint as a multi-label
    classifier (one sigmoid output per product). Requires `transformers`,
    `torch`, and `datasets`.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, DataCollatorWithPadding,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id,
        problem_type="multi_label_classification",
    ).to(device)

    def to_ds(X, y):
        return Dataset.from_dict({
            "text": X.tolist(),
            "label": [_multi_hot(products, label_list) for products in y.tolist()],
        })

    train_ds = to_ds(X_train, y_train)
    test_ds = to_ds(X_test, y_test)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "exact_match": accuracy_score(labels, preds),
        }

    args = TrainingArguments(
        output_dir="bert_extraction_train_tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    fit_seconds = time.perf_counter() - t0

    return {
        "model": model, "tokenizer": tokenizer, "device": device,
        "label_list": label_list, "threshold": threshold,
        "fit_seconds": round(fit_seconds, 2),
    }


def get_or_train_bert_extractor(X_train, y_train, X_test, y_test, label_list,
                                 cache_dir, model_name="distilbert-base-uncased",
                                 epochs=3, batch_size=16, device=None,
                                 threshold=0.5, force_retrain=False):
    """Loads a previously fine-tuned extractor from cache_dir if present;
    otherwise fine-tunes and saves it there for next time."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    marker = os.path.join(cache_dir, "config.json")

    if not force_retrain and os.path.exists(marker):
        print(f"  [cache] loading fine-tuned extractor from {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(cache_dir).to(device)
        return {
            "model": model, "tokenizer": tokenizer, "device": device,
            "label_list": label_list, "threshold": threshold,
            "fit_seconds": 0.0,  # not retrained
        }

    bundle = train_bert_extractor(
        X_train, y_train, X_test, y_test, label_list,
        model_name=model_name, epochs=epochs, batch_size=batch_size,
        device=device, threshold=threshold,
    )

    os.makedirs(cache_dir, exist_ok=True)
    bundle["model"].save_pretrained(cache_dir)
    bundle["tokenizer"].save_pretrained(cache_dir)
    print(f"  [cache] saved fine-tuned extractor to {cache_dir}")

    return bundle


def evaluate_bert_extractor(bundle, X_eval, y_eval) -> dict:
    """
    Runs the fine-tuned multi-label extractor on held-out emails one at a
    time, for an honest per-document latency/throughput measurement.
    Predicted product set = labels whose sigmoid score >= threshold.
    """
    import torch

    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    label_list, threshold = bundle["label_list"], bundle["threshold"]
    model.eval()

    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for text in X_eval.tolist():
            inputs = tokenizer(text, truncation=True, padding=False, max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds.append(sorted(label_list[i] for i, p in enumerate(probs) if p >= threshold))
    elapsed = time.perf_counter() - t0

    return {
        **score_extraction_predictions(preds, y_eval.tolist(), label_list),
        "avg_latency_ms_per_doc": round(elapsed / len(X_eval) * 1000, 3),
        "throughput_docs_per_sec": round(len(X_eval) / elapsed, 1),
        "predictions": preds,
    }


def score_extraction_predictions(preds, y_true, label_list) -> dict:
    """Multi-label scoring shared by the BERT and LLM extraction paths so
    both sides are held to the same metric definitions."""
    y_true_bin = np.array([_multi_hot(p, label_list) for p in y_true])
    y_pred_bin = np.array([_multi_hot(p, label_list) for p in preds])

    per_label_f1 = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)

    return {
        "n": len(y_true),
        "exact_match_ratio": round(accuracy_score(y_true_bin, y_pred_bin), 3),
        "micro_f1": round(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0), 3),
        "macro_f1": round(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0), 3),
        "hamming_loss": round(hamming_loss(y_true_bin, y_pred_bin), 3),
        "per_label_f1": {label_list[i]: round(per_label_f1[i], 3) for i in range(len(label_list))},
    }
