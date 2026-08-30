"""
extraction/llm_extractor.py
-----------------------------
Zero-shot product extraction via the Gemini API (Vertex AI, ADC auth),
plus scoring against ground truth. Mirrors llm_classifier.py's approach
(one call per email, structured JSON response) but asks for a *list* of
products rather than a single label.
"""

import json
import re
import sys
import time

from .. import config
from .bert_extractor import score_extraction_predictions

LLM_PROMPT_TEMPLATE = """You are extracting structured data from a customer support email for a B2B software company.

Identify EVERY product from the following fixed list that this email discusses or references:
{product_labels}

An email may reference zero, one, or multiple products. Only include a product if the email
text actually discusses it — do not guess.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"products": [<zero or more of {product_labels}>]}}

EMAIL:
\"\"\"
{email_text}
\"\"\"
"""


def _parse_llm_json(text: str) -> dict:
    """Strip optional code fences that the model may add despite instructions."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def run_llm_extraction(X_eval, model="gemini-3.5-flash",
                        requests_per_minute=60, max_retries=3):
    """
    Calls the Gemini API zero-shot, once per email, asking it to return the
    set of products (from the fixed taxonomy) the email references. Measures
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
    label_set = set(config.PRODUCT_LABELS)

    preds, latencies_ms, input_tokens, output_tokens = [], [], [], []
    errors = 0

    for i, text in enumerate(X_eval.tolist()):
        prompt = LLM_PROMPT_TEMPLATE.format(
            product_labels=config.PRODUCT_LABELS,
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

        products = parsed.get("products") if parsed else None
        if isinstance(products, list):
            preds.append(sorted(label_set.intersection(products)))
        else:
            preds.append(None)  # parse failure — excluded from accuracy metrics

        if (i + 1) % 20 == 0:
            print(f"  LLM: {i + 1}/{len(X_eval)} emails processed ...")
            time.sleep(delay)

    return {
        "preds": preds,
        "latencies_ms": latencies_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "errors": errors,
        "model": model,
    }


def score_llm_extraction_results(llm_out, y_eval) -> dict:
    preds = llm_out["preds"]
    n_extracted = sum(1 for p in preds if p is not None)

    valid = [i for i in range(len(preds)) if preds[i] is not None]
    y_valid = [y_eval.iloc[i] for i in valid]
    preds_valid = [preds[i] for i in valid]

    total_input_tokens = sum(llm_out["input_tokens"])
    total_output_tokens = sum(llm_out["output_tokens"])
    n = len(preds)

    rates = config.PRICING.get(llm_out["model"])
    real_cost_per_1000 = None
    if rates and n:
        cost_per_email = (
            (total_input_tokens / n / 1_000_000) * rates["input"]
            + (total_output_tokens / n / 1_000_000) * rates["output"]
        )
        real_cost_per_1000 = round(cost_per_email * 1000, 4)

    if valid:
        scores = score_extraction_predictions(preds_valid, y_valid, config.PRODUCT_LABELS)
        scores.pop("n", None)
    else:
        scores = {"exact_match_ratio": None, "micro_f1": None, "macro_f1": None,
                   "hamming_loss": None, "per_label_f1": {}}

    return {
        "model": llm_out["model"],
        "n_attempted": n,
        "n_extracted": n_extracted,
        "n_valid_parses": len(valid),
        "n_errors": llm_out["errors"],
        **scores,
        "avg_latency_ms_per_email": round(sum(llm_out["latencies_ms"]) / n, 1) if n else None,
        "throughput_emails_per_sec_sequential": round(n / (sum(llm_out["latencies_ms"]) / 1000), 2) if n else None,
        "avg_input_tokens_per_email": round(total_input_tokens / n, 1) if n else None,
        "avg_output_tokens_per_email": round(total_output_tokens / n, 1) if n else None,
        "measured_cost_usd_per_1000_emails": real_cost_per_1000,
    }
