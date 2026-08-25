# Email Automation PoC — Fine-Tuned BERT vs. Gemini LLM

Compares a fine-tuned BERT-family transformer against Google Gemini
(zero-shot, via Vertex AI) on the same held-out support emails, for two
tasks: **intent identification** and **criticality classification**.

## Project layout

```
bert-llm-poc/
├── data/
│   └── dataset.csv            # labeled support-email dataset
├── src/
│   └── email_poc/
│       ├── config.py           # constants, .env loading, pricing table
│       ├── dataset.py          # load / profile / split the dataset
│       ├── bert_classifier.py  # fine-tune + evaluate BERT
│       ├── gemini_classifier.py# zero-shot Gemini calls + scoring
│       ├── model_downloader.py # standalone HF checkpoint download helper
│       └── cli.py              # argparse entry point, orchestrates a full run
├── scripts/
│   ├── run_poc.py               # run without installing the package
│   └── download_bert_model.py   # download without installing the package
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Setup

Install as an editable package (recommended — this also registers the
`run-poc` and `download-bert-model` commands):

```bash
pip install -e .
```

Or just install the dependencies and use the `scripts/` entry points instead:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Used by | Purpose |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `email_poc.cli` / `gemini_classifier` | Vertex AI project id |
| `GOOGLE_CLOUD_LOCATION` | `email_poc.cli` / `gemini_classifier` | Vertex AI region |
| `HF_ENDPOINT` | `email_poc.model_downloader` | Internal Artifactory mirror for HuggingFace models |
| `HF_TOKEN` | `email_poc.model_downloader` | Artifactory auth token |
| `REQUESTS_CA_BUNDLE` | `email_poc.model_downloader` | Optional local CA bundle if TLS verification fails behind a corporate proxy |

Then authenticate to Google Cloud with Application Default Credentials (ADC):

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
```

## Dataset

`data/dataset.csv` holds raw support-email threads with these columns (only
a subset is used by the pipeline):

| column | notes |
|---|---|
| `message_body` | raw email text — the model input |
| `email_types` | JSON-list string, e.g. `["issue"]`; first element becomes the `intent` label (`issue` / `inquiry` / `suggestion`) |
| `email_criticality` | `high` / `medium` / `low` |
| `thread_id`, `timestamp` | used to compute reply-latency stats in the dataset profile |

Intent classes with fewer than `MIN_CLASS_SIZE` (3, in `config.py`) examples
are dropped before splitting.

## Running

```bash
# Full run: fine-tune BERT + call Gemini zero-shot on the eval set
run-poc --csv data/dataset.csv
# or, without installing: python scripts/run_poc.py --csv data/dataset.csv

# Faster/cheaper iteration: smaller BERT model, smaller Gemini sample
run-poc --csv data/dataset.csv \
    --bert-model distilbert-base-uncased \
    --bert-epochs 2 \
    --gemini-model gemini-3.1-flash-lite \
    --llm-eval-n 50

# Skip one side if you're iterating on the other
run-poc --csv data/dataset.csv --skip-llm
run-poc --csv data/dataset.csv --skip-bert
```

Key flags: `--bert-model`, `--bert-epochs`, `--bert-batch-size`, `--device`,
`--gemini-model`, `--gemini-rpm` (throttle), `--llm-eval-n` (sample size),
`--force-retrain-bert` (ignore cached fine-tuned weights), `--out` (output
directory, default `poc_outputs/`).

BERT fine-tuned weights are cached per task under `<out>/bert_cache/` and
reloaded on subsequent runs unless `--force-retrain-bert` is passed.

## Outputs (written to `--out`, default `poc_outputs/`)

| file | contents |
|---|---|
| `dataset_profile.json` | descriptive stats: volume, distributions, reply latency |
| `bert_eval_predictions.csv` | eval set + BERT predictions + confidence |
| `gemini_eval_predictions.csv` | eval set (or sample) + Gemini predictions + token usage |
| `poc_results.json` | every headline metric: F1/accuracy, latency, throughput, cost per 1,000 emails |

## model_downloader.py

A standalone helper, independent of the PoC comparison run above: downloads
and caches a HuggingFace checkpoint (`distilbert-base-uncased` by default)
from the company's internal Artifactory mirror to `./models/`, for use in
environments without direct HuggingFace Hub access.

```bash
download-bert-model
# or: python scripts/download_bert_model.py
```

Requires `HF_ENDPOINT` and `HF_TOKEN` in `.env`.

## Notes

- **Pricing**: `PRICING` in `email_poc/config.py` only covers the Gemini
  models selectable via `--gemini-model`; `measured_cost_usd_per_1000_emails`
  is `None` for any model without an entry. Update the rates before
  publishing numbers — check https://ai.google.dev/gemini-api/docs/pricing.
- **Sequential Gemini calls**: `run_gemini_classification` calls the API one
  email at a time (throttled by `--gemini-rpm`) to keep per-call latency
  measurements clean; `throughput_emails_per_sec_sequential` reflects that,
  not a concurrent-request ceiling.
