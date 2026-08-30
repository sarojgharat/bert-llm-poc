"""
model_downloader.py
--------------------
Downloads and caches a HuggingFace checkpoint from the company's internal
Artifactory mirror, for use in environments without direct HuggingFace Hub
access. Independent of the BERT-vs-Gemini comparison pipeline in cli.py.
"""

import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODEL_NAME = "distilbert-base-uncased"
LOCAL_DIR = "./models/distilbert-base-uncased"


def download_model(model_name: str = MODEL_NAME, local_dir: str = LOCAL_DIR) -> None:
    # HF_ENDPOINT, HF_TOKEN, and (optionally) REQUESTS_CA_BUNDLE are read from
    # .env — see .env.example. HF_ENDPOINT points at the company's internal
    # Artifactory mirror for HuggingFace models.
    print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
    print("HF_TOKEN =", bool(os.environ.get("HF_TOKEN")))

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

    print("Downloading model...")
    model = AutoModel.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

    print("Saving locally...")
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"Model saved to {local_dir}")


def main():
    download_model()


if __name__ == "__main__":
    main()
