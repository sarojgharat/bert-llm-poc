"""
bert_classifier.py
-------------------
Fine-tunes and evaluates a HuggingFace sequence-classification model
(intent identification / criticality classification) and derives the
confidence-based BERT-to-LLM escalation curve.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report


def train_bert_classifier(X_train, y_train, X_test, y_test, label_list,
                           model_name="distilbert-base-uncased",
                           epochs=3, batch_size=16, device=None):
    """
    Fine-tunes a real pretrained transformer checkpoint for sequence
    classification. Requires `transformers`, `torch`, and `datasets`,
    plus network access to download the pretrained checkpoint on first run.
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
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)

    def to_ds(X, y):
        return Dataset.from_dict({
            "text": X.tolist(),
            "label": [label2id[l] for l in y.tolist()],
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
        preds = np.argmax(logits, axis=1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }

    args = TrainingArguments(
        output_dir="bert_train_tmp",
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
        "label2id": label2id, "id2label": id2label,
        "fit_seconds": round(fit_seconds, 2),
    }


def get_or_train_bert_classifier(X_train, y_train, X_test, y_test, label_list,
                                  cache_dir, model_name="distilbert-base-uncased",
                                  epochs=3, batch_size=16, device=None,
                                  force_retrain=False):
    """
    Loads a previously fine-tuned model from cache_dir if present;
    otherwise fine-tunes and saves it there for next time.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    marker = os.path.join(cache_dir, "config.json")

    if not force_retrain and os.path.exists(marker):
        print(f"  [cache] loading fine-tuned model from {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(cache_dir).to(device)
        label2id = model.config.label2id
        id2label = model.config.id2label
        return {
            "model": model, "tokenizer": tokenizer, "device": device,
            "label2id": label2id, "id2label": id2label,
            "fit_seconds": 0.0,  # not retrained
        }

    bundle = train_bert_classifier(
        X_train, y_train, X_test, y_test, label_list,
        model_name=model_name, epochs=epochs, batch_size=batch_size, device=device,
    )

    os.makedirs(cache_dir, exist_ok=True)
    bundle["model"].save_pretrained(cache_dir)
    bundle["tokenizer"].save_pretrained(cache_dir)
    print(f"  [cache] saved fine-tuned model to {cache_dir}")

    return bundle


def evaluate_bert_classifier(bundle, X_eval, y_eval, label: str) -> dict:
    """
    Runs the fine-tuned BERT model on held-out emails, one at a time,
    to get an honest per-document latency/throughput measurement (not
    an artificially batched number).
    """
    import torch

    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    id2label = bundle["id2label"]
    model.eval()

    preds, confidences = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for text in X_eval.tolist():
            inputs = tokenizer(text, truncation=True, padding=False, max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            preds.append(id2label[pred_id])
            confidences.append(float(probs[pred_id]))
    elapsed = time.perf_counter() - t0

    macro_f1 = f1_score(y_eval, preds, average="macro")
    weighted_f1 = f1_score(y_eval, preds, average="weighted")
    report = classification_report(y_eval, preds, output_dict=True)

    return {
        "task": label,
        "n": len(y_eval),
        "accuracy": round(accuracy_score(y_eval, preds), 3),
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                          if k not in ("accuracy", "macro avg", "weighted avg")},
        "avg_latency_ms_per_doc": round(elapsed / len(X_eval) * 1000, 3),
        "throughput_docs_per_sec": round(len(X_eval) / elapsed, 1),
        "predictions": preds,
        "confidences": confidences,
    }


def bert_escalation_curve(confidences, preds, y_eval, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Confidence-based escalation curve: how much volume routes to the LLM
    tier at each threshold, and the BERT macro-F1 on what it keeps."""
    conf = np.array(confidences)
    preds = np.array(preds)
    y_eval_arr = pd.Series(y_eval).reset_index(drop=True)

    curve = []
    for t in thresholds:
        keep_mask = conf >= t
        escalate_pct = round((1 - keep_mask.mean()) * 100, 1)
        if keep_mask.sum() > 0:
            f1_retained = f1_score(y_eval_arr[keep_mask], preds[keep_mask], average="macro")
        else:
            f1_retained = None
        curve.append({
            "confidence_threshold": t,
            "pct_escalated_to_llm": escalate_pct,
            "macro_f1_on_retained": round(f1_retained, 3) if f1_retained is not None else None,
        })
    return curve
