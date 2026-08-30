"""
cli.py
------
Command-line entry point: orchestrates dataset loading, BERT fine-tuning +
evaluation, and LLM zero-shot evaluation, then writes the comparison
report used to fill the white paper's placeholder tables.
"""

import argparse
import json
import os

import pandas as pd

from .. import config
from ..dataset import load_dataset, profile_dataset, make_splits
from .bert_classifier import (
    get_or_train_bert_classifier,
    evaluate_bert_classifier,
    bert_escalation_curve,
)
from .llm_classifier import run_llm_classification, score_llm_results
from ..report_utils import SECTION, fmt, fmt_rate, write_comparison_excel


DESCRIPTION = """
Email Automation PoC — Fine-Tuned BERT vs. LLM (Gemini)
=========================================================

Runs the full Proof-of-Concept pipeline referenced in the white paper
"Beyond the Inbox" against a real support-email dataset, comparing:

  A) A genuinely fine-tuned BERT-family transformer (via HuggingFace
     `transformers`) for intent identification and criticality
     classification.
  B) Google Gemini, called zero-shot (no fine-tuning) via Vertex AI,
     on the same held-out emails.

Both sides are scored on the same metrics: F1 (macro/weighted),
per-class F1, latency, throughput, and cost per 1,000 emails.

------------------------------------------------------------------
SETUP
------------------------------------------------------------------
    pip install -e .

    Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in .env (see
    .env.example), then authenticate with Application Default Credentials:

    gcloud auth login
    gcloud config set project <your-project-id>
    gcloud auth application-default login
    gcloud auth application-default set-quota-project <your-project-id>

USAGE
    # Full run: fine-tune BERT + call LLM on the full eval set
    run-classifier-poc --csv data/dataset.csv

    # Faster/cheaper iteration: smaller BERT model, smaller LLM sample
    run-classifier-poc --csv data/dataset.csv \\
        --bert-model distilbert-base-uncased \\
        --bert-epochs 2 \\
        --gemini-model gemini-3.1-flash-lite \\
        --llm-eval-n 50

    # Skip one side if you're iterating on the other
    run-classifier-poc --csv data/dataset.csv --skip-llm
    run-classifier-poc --csv data/dataset.csv --skip-bert

OUTPUTS (written to --out, default ./poc_outputs/)
    dataset_profile.json          - descriptive stats (volume, distributions, reply latency)
    bert_eval_predictions.csv     - eval set + BERT predictions + confidence
    llm_eval_predictions.csv      - eval set (or sample) + LLM predictions + token usage
    poc_results.json              - every headline metric, ready for the white paper tables
    bert_vs_llm_comparison.xlsx   - business-friendly Excel comparison of both models
------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------

def _write_comparison_excel(results: dict, out_path: str) -> None:
    """Write a business-friendly Excel comparison of BERT vs LLM results."""
    bert = results.get("bert", {})
    llm = results.get("llm", {})
    bert_i = bert.get("intent_identification", {})
    bert_c = bert.get("criticality_classification", {})

    _v = fmt
    _rate = fmt_rate

    bert_n = bert_i.get("n")
    llm_n = llm.get("n_attempted")
    bert_fit = bert.get("fit_seconds", {})

    llm_token_detail = "N/A"
    if llm:
        llm_token_detail = (
            f"{_v(llm.get('avg_input_tokens_per_email'))} in / "
            f"{_v(llm.get('avg_output_tokens_per_email'))} out"
        )

    # Each row: (row_type, metric_name, business_description, bert_value, llm_value)
    rows = [
        # ── Intent Identification ──────────────────────────────────────────
        (SECTION, "Intent Identification", "", "", ""),
        ("", "Emails Evaluated",
         "Total number of emails submitted to the model for intent detection",
         _v(bert_n), _v(llm_n)),
        ("", "Intent Successfully Identified",
         "Emails where the model returned a valid intent label (issue / inquiry / suggestion). "
         "BERT always produces a prediction; the LLM may fail to parse some responses.",
         _v(bert_i.get("n_intent_identified")),
         _v(llm.get("n_intent_identified"))),
        ("", "Intent Identification Rate",
         "Percentage of submitted emails where intent was successfully determined",
         _rate(bert_i.get("n_intent_identified"), bert_n),
         _rate(llm.get("n_intent_identified"), llm_n)),
        ("", "Intent Accuracy",
         "Out of correctly-parsed emails, the percentage where the predicted intent matched the ground truth",
         _v(bert_i.get("accuracy"), "pct"),
         _v(llm.get("intent_accuracy"), "pct")),
        ("", "Intent F1 Score (Macro)",
         "Balanced accuracy across all intent classes (0 = worst, 1 = perfect). "
         "Unlike plain accuracy, this penalises the model for being biased toward common classes.",
         _v(bert_i.get("macro_f1")),
         _v(llm.get("intent_macro_f1"))),
        ("", "Intent F1 Score (Weighted)",
         "F1 score weighted by class frequency — closer to what users experience at production volume",
         _v(bert_i.get("weighted_f1")),
         "N/A (not computed for LLM)"),

        # ── Criticality Classification ─────────────────────────────────────
        (SECTION, "Criticality Classification", "", "", ""),
        ("", "Emails Evaluated",
         "Total number of emails submitted for urgency classification",
         _v(bert_n), _v(llm_n)),
        ("", "Criticality Successfully Classified",
         "Emails where the model returned a valid urgency label (high / medium / low). "
         "BERT always produces a prediction; the LLM may fail to parse some responses.",
         _v(bert_c.get("n_criticality_classified")),
         _v(llm.get("n_criticality_classified"))),
        ("", "Criticality Classification Rate",
         "Percentage of submitted emails where urgency was successfully determined",
         _rate(bert_c.get("n_criticality_classified"), bert_n),
         _rate(llm.get("n_criticality_classified"), llm_n)),
        ("", "Criticality Accuracy",
         "Out of correctly-parsed emails, the percentage where the predicted urgency matched the ground truth",
         _v(bert_c.get("accuracy"), "pct"),
         _v(llm.get("criticality_accuracy"), "pct")),
        ("", "Criticality F1 Score (Macro)",
         "Balanced accuracy across all urgency levels (0 = worst, 1 = perfect). "
         "Accounts for class imbalance — important if 'high' urgency emails are rare.",
         _v(bert_c.get("macro_f1")),
         _v(llm.get("criticality_macro_f1"))),
        ("", "Criticality F1 Score (Weighted)",
         "F1 score weighted by class frequency — closer to what users experience at production volume",
         _v(bert_c.get("weighted_f1")),
         "N/A (not computed for LLM)"),

        # ── Speed & Cost ───────────────────────────────────────────────────
        (SECTION, "Speed & Cost", "", "", ""),
        ("", "Avg Response Time per Email",
         "Average wall-clock time to classify a single email. Lower is faster.",
         _v(bert_i.get("avg_latency_ms_per_doc"), "ms"),
         _v(llm.get("avg_latency_ms_per_email"), "ms")),
        ("", "Throughput (emails / sec, sequential)",
         "How many emails per second are processed one-at-a-time. "
         "Higher throughput = lower infrastructure footprint for the same volume.",
         _v(bert_i.get("throughput_docs_per_sec")),
         _v(llm.get("throughput_emails_per_sec_sequential"))),
        ("", "Estimated Cost per 1,000 Emails",
         "API or compute cost to classify 1,000 emails. "
         "BERT incurs a one-time training cost; ongoing inference is near-zero on local hardware.",
         "~$0.00 (near-zero post-training)",
         _v(llm.get("measured_cost_usd_per_1000_emails"), "usd") if llm.get("measured_cost_usd_per_1000_emails") is not None else "N/A"),
        ("", "Avg Tokens per Email (Input / Output)",
         "LLM-only metric: token volume consumed per email. "
         "This directly drives the per-email API cost.",
         "N/A (BERT is not token-based at inference)",
         llm_token_detail),

        # ── Setup & Operability ────────────────────────────────────────────
        (SECTION, "Setup & Operability", "", "", ""),
        ("", "Model",
         "Identifier of the model evaluated",
         bert.get("model_name", "N/A"),
         _v(llm.get("model"))),
        ("", "Requires Labeled Training Data",
         "Does the model need human-annotated examples before it can classify emails?",
         "Yes — needs annotated email dataset",
         "No — works out of the box (zero-shot)"),
        ("", "One-time Training Time",
         "Time spent fine-tuning BERT on labeled examples; zero for the LLM approach. "
         "Intent and criticality are trained as separate models.",
         (f"Intent: {_v(bert_fit.get('intent'))}s  |  "
          f"Criticality: {_v(bert_fit.get('criticality'))}s"),
         "0 s (no training required)"),
        ("", "External API Required at Runtime",
         "Does classifying an email require an outbound API call (latency & cost implications)?",
         "No — model runs locally after training",
         "Yes — each email calls the Gemini API"),
        ("", "LLM Parse / Format Errors",
         "Emails where the LLM response was malformed and could not be parsed. "
         "BERT always returns a structured prediction.",
         "0",
         _v(llm.get("n_errors"))),
        ("", "LLM Valid Joint Parses",
         "Emails where BOTH intent and criticality were successfully parsed from the LLM response "
         "(used as the denominator for LLM accuracy metrics)",
         _v(bert_n),
         _v(llm.get("n_valid_parses"))),
    ]

    write_comparison_excel(rows, out_path, "BERT vs LLM Comparison", "BERT", f"LLM ({llm.get('model', 'Gemini')})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=config.DEFAULT_DATA_PATH, help="Path to the labeled email dataset CSV")
    parser.add_argument("--out", default="poc_outputs", help="Output directory")

    parser.add_argument("--bert-model", default="distilbert-base-uncased",
                         help="Any HuggingFace sequence-classification checkpoint, e.g. bert-base-uncased")
    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (auto-detected if omitted)")

    # Note: measured_cost_usd_per_1000_emails is only populated for models with
    # an entry in config.PRICING; others report cost as None.
    parser.add_argument("--gemini-model", default="gemini-3.5-flash",
                         choices=["gemma-4-31b-it", "gemini-3.1-flash-lite", "gemini-3-flash",
                                  "gemini-3.5-flash-001", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                                  "gemini-3.1-pro", "gemini-3.1-pro-preview"])
    parser.add_argument("--gemini-rpm", type=int, default=60, help="Requests per minute throttle")
    parser.add_argument("--llm-eval-n", type=int, default=None,
                         help="How many eval emails to send to the LLM (default: all of the eval set)")

    parser.add_argument("--force-retrain-bert", default=False, action="store_true",
                         help="Ignore any cached fine-tuned BERT model and retrain from scratch")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = load_dataset(args.csv)

    print("Profiling dataset ...")
    profile = profile_dataset(df)
    with open(os.path.join(args.out, "dataset_profile.json"), "w") as f:
        json.dump(profile, f, indent=2, default=str)

    print("Building 80/10/10 stratified split ...")
    splits = make_splits(df)
    X_train, yi_train, yc_train = splits["train"]
    X_test, yi_test, yc_test = splits["test"]
    X_eval, yi_eval, yc_eval = splits["eval"]

    results = {"dataset_profile": profile}

    # ── BERT ─────────────────────────────────────────────────────────────
    if not args.skip_bert:
        cache_root = os.path.join(args.out, "bert_cache")
        print(f"\nLoading/fine-tuning {args.bert_model} for intent identification ...")
        bundle_i = get_or_train_bert_classifier(
            X_train, yi_train, X_test, yi_test, config.INTENT_LABELS,
            cache_dir=os.path.join(cache_root, "intent"),
            model_name=args.bert_model, epochs=args.bert_epochs,
            batch_size=args.bert_batch_size, device=args.device,
            force_retrain=args.force_retrain_bert,
        )
        print(f"Loading/fine-tuning {args.bert_model} for criticality classification ...")
        bundle_c = get_or_train_bert_classifier(
            X_train, yc_train, X_test, yc_test, config.CRITICALITY_LABELS,
            cache_dir=os.path.join(cache_root, "criticality"),
            model_name=args.bert_model, epochs=args.bert_epochs,
            batch_size=args.bert_batch_size, device=args.device,
            force_retrain=args.force_retrain_bert,
        )

        print("Evaluating BERT on held-out set ...")
        result_i = evaluate_bert_classifier(bundle_i, X_eval, yi_eval, "intent_identification")
        result_c = evaluate_bert_classifier(bundle_c, X_eval, yc_eval, "criticality_classification")

        curve_i = bert_escalation_curve(result_i["confidences"], result_i["predictions"], yi_eval)
        curve_c = bert_escalation_curve(result_c["confidences"], result_c["predictions"], yc_eval)

        pd.DataFrame({
            "message_body": X_eval,
            "true_intent": yi_eval,
            "pred_intent": result_i["predictions"],
            "intent_confidence": result_i["confidences"],
            "true_criticality": yc_eval,
            "pred_criticality": result_c["predictions"],
            "criticality_confidence": result_c["confidences"],
        }).to_csv(os.path.join(args.out, "bert_eval_predictions.csv"))

        results["bert"] = {
            "model_name": args.bert_model,
            "fit_seconds": {
                "intent": bundle_i["fit_seconds"],
                "criticality": bundle_c["fit_seconds"],
            },
            "intent_identification": {
                "n_intent_identified": len(result_i["predictions"]),  # BERT always predicts all
                **{k: v for k, v in result_i.items() if k not in ("predictions", "confidences")},
            },
            "criticality_classification": {
                "n_criticality_classified": len(result_c["predictions"]),  # BERT always predicts all
                **{k: v for k, v in result_c.items() if k not in ("predictions", "confidences")},
            },
            "confidence_escalation_curve": {"intent": curve_i, "criticality": curve_c},
        }
    else:
        print("Skipping BERT fine-tuning (--skip-bert).")

    # ── LLM ──────────────────────────────────────────────────────────────
    if not args.skip_llm:
        n = args.llm_eval_n or len(X_eval)
        n = min(n, len(X_eval))
        print(f"\nCalling LLM ({args.gemini_model}) zero-shot on {n} eval emails ...")
        X_llm = X_eval.iloc[:n].reset_index(drop=True)
        yi_llm = yi_eval.iloc[:n].reset_index(drop=True)
        yc_llm = yc_eval.iloc[:n].reset_index(drop=True)

        llm_out = run_llm_classification(
            X_llm, model=args.gemini_model,
            requests_per_minute=args.gemini_rpm,
        )
        llm_scores = score_llm_results(llm_out, yi_llm, yc_llm)

        pd.DataFrame({
            "message_body": X_llm,
            "true_intent": yi_llm,
            "pred_intent": llm_out["preds_intent"],
            "true_criticality": yc_llm,
            "pred_criticality": llm_out["preds_criticality"],
            "latency_ms": llm_out["latencies_ms"],
            "input_tokens": llm_out["input_tokens"],
            "output_tokens": llm_out["output_tokens"],
        }).to_csv(os.path.join(args.out, "llm_eval_predictions.csv"))

        results["llm"] = llm_scores
    else:
        print("Skipping LLM evaluation (--skip-llm).")

    # ── Write JSON results ────────────────────────────────────────────────
    json_path = os.path.join(args.out, "poc_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Write Excel comparison ────────────────────────────────────────────
    excel_path = os.path.join(args.out, "bert_vs_llm_comparison.xlsx")
    _write_comparison_excel(results, excel_path)

    print("\n=== SUMMARY ===")
    print(json.dumps(results.get("bert", {}), indent=2, default=str)[:1500])
    print(json.dumps(results.get("llm", {}), indent=2, default=str))
    print(f"\nFull results written to {json_path}")


if __name__ == "__main__":
    main()
