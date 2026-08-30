"""
config.py
---------
Shared constants and environment configuration for the BERT-vs-Gemini PoC.
Loads .env once on first import so every other module can read os.environ
without repeating the load_dotenv() call.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dataset / splitting
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join("data", "dataset.csv")

RANDOM_STATE = 42
TRAIN_SIZE, TEST_SIZE, EVAL_SIZE = 0.80, 0.10, 0.10   # of full dataset
MIN_CLASS_SIZE = 3                                    # drop intent classes smaller than this

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
INTENT_LABELS = ["issue", "inquiry", "suggestion"]
CRITICALITY_LABELS = ["high", "medium", "low"]

# Canonical product taxonomy for the data-extraction comparison (see
# email_poc.extraction). The raw `product_types` column has inconsistent
# casing (e.g. "IAM service" vs "IAM Service") which is normalized against
# this list when the extraction dataset is loaded.
PRODUCT_LABELS = ["API Development", "API Monitoring", "Cloud Management", "IAM Service", "Mercury Language"]

# ---------------------------------------------------------------------------
# Gemini / Vertex AI
# ---------------------------------------------------------------------------
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "bert-vs-llm-poc")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west2")

# Real (July 2026) Gemini API pricing, USD per million tokens.
# Update these if pricing changes: https://ai.google.dev/gemini-api/docs/pricing
# Only models listed in cli.py's --gemini-model choices need an entry here.
PRICING = {
    "gemini-3.1-flash-lite":  {"input": 0.25, "output": 1.50},
    "gemini-3-flash":         {"input": 0.50, "output": 3.00},
    "gemini-3.5-flash":       {"input": 1.50, "output": 9.00},
    "gemini-3.1-pro":         {"input": 2.00, "output": 12.00},
}
CHARS_PER_TOKEN = 4.0  # rough heuristic, only used for the dataset profile step
