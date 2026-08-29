#!/usr/bin/env python
"""Run the classifier PoC without installing the package (`pip install -e .` not required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from email_poc.classifier.cli import main  # noqa: E402
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Apply corporate CA bundle before any Google/gRPC library initialises.
# GRPC_DEFAULT_SSL_ROOTS_FILE_PATH is read by the gRPC C extension at first import,
# so it must be set here — before vertexai / google-cloud-aiplatform are imported.
_ca = os.getenv("CUSTOM_CA_CERTS_FILE", "")
if _ca:
    os.environ["REQUESTS_CA_BUNDLE"] = _ca
    os.environ["SSL_CERT_FILE"] = _ca
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _ca

if __name__ == "__main__":
    main()
