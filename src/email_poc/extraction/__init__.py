"""Data-extraction comparison: fine-tuned BERT vs. zero-shot LLM at
extracting the set of products referenced in a support email.

Kept independent from the intent/criticality classification comparison
in email_poc.cli — separate dataset prep, model code, and CLI, sharing
only email_poc.config and email_poc.report_utils.
"""
