"""
dataset.py
----------
Loading, profiling, and train/test/eval splitting of the raw email dataset.
"""

import ast

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["intent"] = df["email_types"].apply(lambda x: ast.literal_eval(x)[0])
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Descriptive stats computed directly from the data (no invented numbers)."""
    threads = df.sort_values(["thread_id", "timestamp"]).copy()
    threads["delta_hours"] = (
        threads.groupby("thread_id")["timestamp"].diff().dt.total_seconds() / 3600
    )
    reply_gaps = threads["delta_hours"].dropna()

    return {
        "n_emails": len(df),
        "n_threads": df["thread_id"].nunique(),
        "avg_messages_per_thread": round(df.groupby("thread_id").size().mean(), 2),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "avg_message_chars": round(df["message_body"].str.len().mean(), 1),
        "avg_message_tokens_est": round(df["message_body"].str.len().mean() / config.CHARS_PER_TOKEN, 1),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "criticality_distribution": df["email_criticality"].value_counts().to_dict(),
        "status_distribution": df["email_status"].value_counts().to_dict(),
        "reply_latency_hours": {
            "n": int(reply_gaps.shape[0]),
            "mean": round(reply_gaps.mean(), 2),
            "median": round(reply_gaps.median(), 2),
            "p95": round(reply_gaps.quantile(0.95), 2),
        },
    }


def make_splits(df: pd.DataFrame):
    counts = df["intent"].value_counts()
    keep_classes = counts[counts >= config.MIN_CLASS_SIZE].index
    df = df[df["intent"].isin(keep_classes)].copy().reset_index(drop=True)

    X, y_intent, y_crit = df["message_body"], df["intent"], df["email_criticality"]

    X_train, X_temp, yi_train, yi_temp, yc_train, yc_temp = train_test_split(
        X, y_intent, y_crit,
        test_size=(1 - config.TRAIN_SIZE), random_state=config.RANDOM_STATE, stratify=y_intent,
    )
    rel_test = config.TEST_SIZE / (config.TEST_SIZE + config.EVAL_SIZE)
    X_test, X_eval, yi_test, yi_eval, yc_test, yc_eval = train_test_split(
        X_temp, yi_temp, yc_temp,
        test_size=(1 - rel_test), random_state=config.RANDOM_STATE, stratify=yi_temp,
    )
    return {
        "train": (X_train.reset_index(drop=True), yi_train.reset_index(drop=True), yc_train.reset_index(drop=True)),
        "test": (X_test.reset_index(drop=True), yi_test.reset_index(drop=True), yc_test.reset_index(drop=True)),
        "eval": (X_eval.reset_index(drop=True), yi_eval.reset_index(drop=True), yc_eval.reset_index(drop=True)),
    }
