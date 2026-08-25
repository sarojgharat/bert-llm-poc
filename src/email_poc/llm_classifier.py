"""
llm_classifier.py
-----------------
Zero-shot email classification via the Gemini API (Vertex AI, ADC auth),
plus scoring against ground truth.
"""

import json
import re
import sys
import time

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from . import config


LLM_PROMPT_TEMPLATE = """You are classifying a customer support email for a B2B software company.

Classify the email on TWO dimensions:

1. intent — exactly one of: {intent_labels}
   - "issue": the sender is reporting a problem or defect, or this is a reply handling one
   - "inquiry": the sender is asking a question or requesting information
   - "suggestion": the sender is proposing a feature or improvement, or this is a reply to one

2. criticality — exactly one of: {criticality_labels}
   - "high": urgent, business-impacting, production-down language
   - "medium": real but not urgent/blocking
   - "low": routine, no urgency indicated

Respond with ONLY a JSON object, no other text, in this exact format:
{{"intent": "<one of {intent_labels}>", "criticality": "<one of {criticality_labels}>"}}

EMAIL:
\"\"\"
{email_text}
\"\"\"
"""


def _parse_llm_json(text: str) -> dict:
    """Strip optional code fences that the model may add despite instructions."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def run_llm_classification(X_eval, model="gemini-3.5-flash",
                            requests_per_minute=60, max_retries=3):
    """
    Calls the Gemini API zero-shot, once per email, asking for both
    intent and criticality in a single structured-JSON response. Measures
    real per-call latency and real token usage (for cost calculation).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )

    delay = 60.0 / requests_per_minute

    preds_intent, preds_crit = [], []
    latencies_ms, input_tokens, output_tokens = [], [], []
    errors = 0

    for i, text in enumerate(X_eval.tolist()):
        prompt = LLM_PROMPT_TEMPLATE.format(
            intent_labels=config.INTENT_LABELS,
            criticality_labels=config.CRITICALITY_LABELS,
            email_text=text[:4000],
        )
        parsed = None
        t0 = time.perf_counter()
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                parsed = _parse_llm_json(response.text)
                usage = getattr(response, "usage_metadata", None)
                input_tokens.append(getattr(usage, "prompt_token_count", 0) if usage else 0)
                output_tokens.append(getattr(usage, "candidates_token_count", 0) if usage else 0)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == max_retries - 1:
                    print(f"  [warn] email {i}: giving up after {max_retries} attempts ({e})", file=sys.stderr)
                    errors += 1
                    input_tokens.append(0)
                    output_tokens.append(0)
                else:
                    time.sleep(delay * (attempt + 1))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        if parsed and parsed.get("intent") in config.INTENT_LABELS:
            preds_intent.append(parsed["intent"])
        else:
            preds_intent.append(None)
        if parsed and parsed.get("criticality") in config.CRITICALITY_LABELS:
            preds_crit.append(parsed["criticality"])
        else:
            preds_crit.append(None)

        if (i + 1) % 20 == 0:
            print(f"  LLM: {i + 1}/{len(X_eval)} emails classified ...")
            time.sleep(delay)

    return {
        "preds_intent": preds_intent,
        "preds_criticality": preds_crit,
        "latencies_ms": latencies_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "errors": errors,
        "model": model,
    }


def score_llm_results(llm_out, yi_eval, yc_eval) -> dict:
    preds_i = llm_out["preds_intent"]
    preds_c = llm_out["preds_criticality"]

    n_intent_identified = sum(1 for p in preds_i if p is not None)
    n_criticality_classified = sum(1 for p in preds_c if p is not None)

    # Drop rows where parsing failed on either dimension for a fair joint score.
    valid = [i for i in range(len(preds_i)) if preds_i[i] is not None and preds_c[i] is not None]
    yi = pd.Series(yi_eval).reset_index(drop=True)
    yc = pd.Series(yc_eval).reset_index(drop=True)

    yi_valid = yi.iloc[valid]
    yc_valid = yc.iloc[valid]
    preds_i_valid = [preds_i[i] for i in valid]
    preds_c_valid = [preds_c[i] for i in valid]

    total_input_tokens = sum(llm_out["input_tokens"])
    total_output_tokens = sum(llm_out["output_tokens"])
    n = len(preds_i)

    rates = config.PRICING.get(llm_out["model"])
    real_cost_per_1000 = None
    if rates and n:
        cost_per_email = (
            (total_input_tokens / n / 1_000_000) * rates["input"]
            + (total_output_tokens / n / 1_000_000) * rates["output"]
        )
        real_cost_per_1000 = round(cost_per_email * 1000, 4)

    return {
        "model": llm_out["model"],
        "n_attempted": n,
        "n_intent_identified": n_intent_identified,
        "n_criticality_classified": n_criticality_classified,
        "n_valid_parses": len(valid),
        "n_errors": llm_out["errors"],
        "intent_macro_f1": round(f1_score(yi_valid, preds_i_valid, average="macro"), 3) if valid else None,
        "intent_accuracy": round(accuracy_score(yi_valid, preds_i_valid), 3) if valid else None,
        "criticality_macro_f1": round(f1_score(yc_valid, preds_c_valid, average="macro"), 3) if valid else None,
        "criticality_accuracy": round(accuracy_score(yc_valid, preds_c_valid), 3) if valid else None,
        "avg_latency_ms_per_email": round(sum(llm_out["latencies_ms"]) / n, 1) if n else None,
        "throughput_emails_per_sec_sequential": round(n / (sum(llm_out["latencies_ms"]) / 1000), 2) if n else None,
        "avg_input_tokens_per_email": round(total_input_tokens / n, 1) if n else None,
        "avg_output_tokens_per_email": round(total_output_tokens / n, 1) if n else None,
        "measured_cost_usd_per_1000_emails": real_cost_per_1000,
    }
