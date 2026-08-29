"""
extraction/cli.py
-------------------
Command-line entry point for the data-extraction comparison: orchestrates
dataset loading, BERT multi-label fine-tuning + evaluation, and LLM
zero-shot extraction, then writes the comparison report. Independent of
email_poc.cli (the intent/criticality classification comparison) — the
two pipelines share only email_poc.config and email_poc.report_utils.
"""

import argparse
import json
import os

import pandas as pd

from .. import config
from ..report_utils import SECTION, fmt, fmt_rate, write_comparison_excel
from .dataset import load_extraction_dataset, make_extraction_splits
from .bert_extractor import get_or_train_bert_extractor, evaluate_bert_extractor
from .llm_extractor import run_llm_extraction, score_llm_extraction_results


DESCRIPTION = """
Email Automation PoC — Data Extraction: Fine-Tuned BERT vs. LLM (Gemini)
==========================================================================

Runs the product-extraction PoC against the same support-email dataset
used for the intent/criticality comparison (email_poc.cli), comparing:

  A) A fine-tuned BERT-family transformer, used as a multi-label
     classifier over the known product taxonomy (an email can reference
     more than one product).
  B) Google Gemini, called zero-shot (no fine-tuning) via Vertex AI,
     asked to return the same set of products per email.

Both sides are scored on the same metrics: exact-match ratio, micro/macro
F1, per-product F1, latency, throughput, and cost per 1,000 emails.

------------------------------------------------------------------
USAGE
    # Full run
    run-extraction-poc --csv data/dataset.csv

    # Faster/cheaper iteration
    run-extraction-poc --csv data/dataset.csv \\
        --bert-model distilbert-base-uncased \\
        --bert-epochs 2 \\
        --gemini-model gemini-3.1-flash-lite \\
        --llm-eval-n 50

    # Skip one side if you're iterating on the other
    run-extraction-poc --csv data/dataset.csv --skip-llm
    run-extraction-poc --csv data/dataset.csv --skip-bert

OUTPUTS (written to --out, default ./poc_outputs/)
    bert_extraction_eval_predictions.csv  - eval set + BERT predicted product sets
    llm_extraction_eval_predictions.csv   - eval set (or sample) + LLM predicted product sets
    extraction_results.json               - every headline metric
    bert_vs_llm_extraction_comparison.xlsx - business-friendly Excel comparison
------------------------------------------------------------------
"""


def _write_extraction_comparison_excel(results: dict, out_path: str) -> None:
    bert = results.get("bert", {})
    llm = results.get("llm", {})

    bert_n = bert.get("n")
    llm_n = llm.get("n_attempted")

    llm_token_detail = "N/A"
    if llm:
        llm_token_detail = (
            f"{fmt(llm.get('avg_input_tokens_per_email'))} in / "
            f"{fmt(llm.get('avg_output_tokens_per_email'))} out"
        )

    per_label_rows = []
    all_labels = sorted(set(bert.get("per_label_f1", {})) | set(llm.get("per_label_f1", {})))
    for label in all_labels:
        per_label_rows.append((
            "", f"Per-Product F1 — {label}",
            f"F1 score for correctly identifying (or correctly excluding) '{label}'",
            fmt(bert.get("per_label_f1", {}).get(label)),
            fmt(llm.get("per_label_f1", {}).get(label)),
        ))

    rows = [
        (SECTION, "Product Extraction", "", "", ""),
        ("", "Emails Evaluated",
         "Total number of emails submitted for product extraction",
         fmt(bert_n), fmt(llm_n)),
        ("", "Emails Successfully Processed",
         "Emails where the model returned a usable product list. "
         "BERT always produces a prediction; the LLM may fail to parse some responses.",
         fmt(bert_n), fmt(llm.get("n_valid_parses"))),
        ("", "Processing Success Rate",
         "Percentage of submitted emails where a product list was successfully extracted",
         fmt_rate(bert_n, bert_n), fmt_rate(llm.get("n_valid_parses"), llm_n)),
        ("", "Exact Match Ratio",
         "Out of correctly-parsed emails, the percentage where the predicted product set "
         "matched the ground-truth set exactly (all products right, none missing, none extra)",
         fmt(bert.get("exact_match_ratio"), "pct"),
         fmt(llm.get("exact_match_ratio"), "pct")),
        ("", "Micro F1",
         "F1 computed over all (email, product) pairs pooled together — dominated by common products",
         fmt(bert.get("micro_f1")), fmt(llm.get("micro_f1"))),
        ("", "Macro F1",
         "F1 averaged per product then across products — treats rare and common products equally",
         fmt(bert.get("macro_f1")), fmt(llm.get("macro_f1"))),
        ("", "Hamming Loss",
         "Fraction of individual product labels (present/absent) predicted incorrectly, "
         "across all products and emails. Lower is better.",
         fmt(bert.get("hamming_loss")), fmt(llm.get("hamming_loss"))),

        (SECTION, "Per-Product Breakdown", "", "", ""),
        *per_label_rows,

        (SECTION, "Speed & Cost", "", "", ""),
        ("", "Avg Response Time per Email",
         "Average wall-clock time to extract products from a single email. Lower is faster.",
         fmt(bert.get("avg_latency_ms_per_doc"), "ms"),
         fmt(llm.get("avg_latency_ms_per_email"), "ms")),
        ("", "Throughput (emails / sec, sequential)",
         "How many emails per second are processed one-at-a-time. "
         "Higher throughput = lower infrastructure footprint for the same volume.",
         fmt(bert.get("throughput_docs_per_sec")),
         fmt(llm.get("throughput_emails_per_sec_sequential"))),
        ("", "Estimated Cost per 1,000 Emails",
         "API or compute cost to process 1,000 emails. "
         "BERT incurs a one-time training cost; ongoing inference is near-zero on local hardware.",
         "~$0.00 (near-zero post-training)",
         fmt(llm.get("measured_cost_usd_per_1000_emails"), "usd") if llm.get("measured_cost_usd_per_1000_emails") is not None else "N/A"),
        ("", "Avg Tokens per Email (Input / Output)",
         "LLM-only metric: token volume consumed per email. This directly drives the per-email API cost.",
         "N/A (BERT is not token-based at inference)",
         llm_token_detail),

        (SECTION, "Setup & Operability", "", "", ""),
        ("", "Model",
         "Identifier of the model evaluated",
         bert.get("model_name", "N/A"), fmt(llm.get("model"))),
        ("", "Requires Labeled Training Data",
         "Does the model need human-annotated examples before it can extract products?",
         "Yes — needs annotated email dataset",
         "No — works out of the box (zero-shot)"),
        ("", "One-time Training Time",
         "Time spent fine-tuning the multi-label BERT classifier; zero for the LLM approach.",
         f"{fmt(bert.get('fit_seconds'))}s", "0 s (no training required)"),
        ("", "External API Required at Runtime",
         "Does extracting from an email require an outbound API call (latency & cost implications)?",
         "No — model runs locally after training",
         "Yes — each email calls the Gemini API"),
        ("", "LLM Parse / Format Errors",
         "Emails where the LLM response was malformed and could not be parsed. "
         "BERT always returns a structured prediction.",
         "0", fmt(llm.get("n_errors"))),
    ]

    write_comparison_excel(rows, out_path, "BERT vs LLM Extraction",
                            "BERT", f"LLM ({llm.get('model', 'Gemini')})")


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=config.DEFAULT_DATA_PATH, help="Path to the labeled email dataset CSV")
    parser.add_argument("--out", default="poc_outputs", help="Output directory")

    parser.add_argument("--bert-model", default="distilbert-base-uncased",
                         help="Any HuggingFace sequence-classification checkpoint, e.g. bert-base-uncased")
    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-threshold", type=float, default=0.5,
                         help="Sigmoid threshold above which a product is predicted present")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (auto-detected if omitted)")

    parser.add_argument("--gemini-model", default="gemini-3.5-flash",
                         choices=["gemma-4-31b-it", "gemini-3.1-flash-lite", "gemini-3-flash",
                                  "gemini-3.5-flash-001", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                                  "gemini-3.1-pro", "gemini-3.1-pro-preview"])
    parser.add_argument("--gemini-rpm", type=int, default=60, help="Requests per minute throttle")
    parser.add_argument("--llm-eval-n", type=int, default=None,
                         help="How many eval emails to send to the LLM (default: all of the eval set)")

    parser.add_argument("--force-retrain-bert", default=False, action="store_true",
                         help="Ignore any cached fine-tuned BERT extractor and retrain from scratch")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = load_extraction_dataset(args.csv)

    print("Building 80/10/10 stratified split ...")
    splits = make_extraction_splits(df)
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    X_eval, y_eval = splits["eval"]

    results = {}

    # ── BERT ─────────────────────────────────────────────────────────────
    if not args.skip_bert:
        cache_dir = os.path.join(args.out, "bert_extraction_cache")
        print(f"\nLoading/fine-tuning {args.bert_model} for product extraction ...")
        bundle = get_or_train_bert_extractor(
            X_train, y_train, X_test, y_test, config.PRODUCT_LABELS,
            cache_dir=cache_dir,
            model_name=args.bert_model, epochs=args.bert_epochs,
            batch_size=args.bert_batch_size, device=args.device,
            threshold=args.bert_threshold, force_retrain=args.force_retrain_bert,
        )

        print("Evaluating BERT extractor on held-out set ...")
        result = evaluate_bert_extractor(bundle, X_eval, y_eval)

        pd.DataFrame({
            "message_body": X_eval,
            "true_products": y_eval,
            "pred_products": result["predictions"],
        }).to_csv(os.path.join(args.out, "bert_extraction_eval_predictions.csv"))

        results["bert"] = {
            "model_name": args.bert_model,
            "fit_seconds": bundle["fit_seconds"],
            **{k: v for k, v in result.items() if k != "predictions"},
        }
    else:
        print("Skipping BERT fine-tuning (--skip-bert).")

    # ── LLM ──────────────────────────────────────────────────────────────
    if not args.skip_llm:
        n = args.llm_eval_n or len(X_eval)
        n = min(n, len(X_eval))
        print(f"\nCalling LLM ({args.gemini_model}) zero-shot on {n} eval emails ...")
        X_llm = X_eval.iloc[:n].reset_index(drop=True)
        y_llm = y_eval.iloc[:n].reset_index(drop=True)

        llm_out = run_llm_extraction(
            X_llm, model=args.gemini_model,
            requests_per_minute=args.gemini_rpm,
        )
        llm_scores = score_llm_extraction_results(llm_out, y_llm)

        pd.DataFrame({
            "message_body": X_llm,
            "true_products": y_llm,
            "pred_products": llm_out["preds"],
            "latency_ms": llm_out["latencies_ms"],
            "input_tokens": llm_out["input_tokens"],
            "output_tokens": llm_out["output_tokens"],
        }).to_csv(os.path.join(args.out, "llm_extraction_eval_predictions.csv"))

        results["llm"] = llm_scores
    else:
        print("Skipping LLM evaluation (--skip-llm).")

    # ── Write JSON results ────────────────────────────────────────────────
    json_path = os.path.join(args.out, "extraction_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Write Excel comparison ────────────────────────────────────────────
    excel_path = os.path.join(args.out, "bert_vs_llm_extraction_comparison.xlsx")
    _write_extraction_comparison_excel(results, excel_path)

    print("\n=== SUMMARY ===")
    print(json.dumps(results.get("bert", {}), indent=2, default=str)[:1500])
    print(json.dumps(results.get("llm", {}), indent=2, default=str))
    print(f"\nFull results written to {json_path}")


if __name__ == "__main__":
    main()
