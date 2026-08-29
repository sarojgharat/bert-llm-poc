"""
extraction/dataset.py
----------------------
Loading and splitting the email dataset for the product-extraction
comparison: given the email body, extract *which* of the known products
(possibly more than one) it discusses.

Ground truth comes from the raw `product_types` column, which is a
JSON-list string with inconsistent casing (e.g. "IAM service" vs
"IAM Service"). This module normalizes it against config.PRODUCT_LABELS.
"""

import ast

import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config
from ..dataset import load_dataset as load_raw_dataset

_CANONICAL_BY_LOWER = {label.lower(): label for label in config.PRODUCT_LABELS}


def _normalize_products(raw: str) -> list:
    """Parse the JSON-list string and map each entry to its canonical label,
    dropping anything outside the known taxonomy."""
    values = ast.literal_eval(raw)
    normalized = {_CANONICAL_BY_LOWER[v.strip().lower()]
                  for v in values if v.strip().lower() in _CANONICAL_BY_LOWER}
    return sorted(normalized)


def load_extraction_dataset(csv_path: str) -> pd.DataFrame:
    df = load_raw_dataset(csv_path)
    df["products"] = df["product_types"].apply(_normalize_products)
    return df[df["products"].apply(len) > 0].reset_index(drop=True)


def make_extraction_splits(df: pd.DataFrame):
    """
    80/10/10 split, stratified on each email's "primary product" (the
    alphabetically-first product in its ground-truth set). The full
    product *combination* (e.g. ("Cloud Management", "IAM Service")) has
    ~25 distinct values, several with fewer than 10 rows — too fine-grained
    to survive being stratified across three splits. The 5-way primary-product
    split keeps every split reasonably balanced across the taxonomy without
    that fragility.
    """
    primary = df["products"].apply(lambda products: sorted(products)[0])
    counts = primary.value_counts()
    keep = counts[counts >= config.MIN_CLASS_SIZE].index
    mask = primary.isin(keep)
    df = df[mask].copy().reset_index(drop=True)
    primary = primary[mask].reset_index(drop=True)

    X, y = df["message_body"], df["products"]

    X_train, X_temp, y_train, y_temp, primary_train, primary_temp = train_test_split(
        X, y, primary,
        test_size=(1 - config.TRAIN_SIZE), random_state=config.RANDOM_STATE, stratify=primary,
    )
    rel_test = config.TEST_SIZE / (config.TEST_SIZE + config.EVAL_SIZE)
    X_test, X_eval, y_test, y_eval, _, _ = train_test_split(
        X_temp, y_temp, primary_temp,
        test_size=(1 - rel_test), random_state=config.RANDOM_STATE, stratify=primary_temp,
    )
    return {
        "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        "eval": (X_eval.reset_index(drop=True), y_eval.reset_index(drop=True)),
    }
