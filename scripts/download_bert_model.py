#!/usr/bin/env python
"""Download the BERT checkpoint without installing the package (`pip install -e .` not required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from email_poc.model_downloader import main  # noqa: E402

if __name__ == "__main__":
    main()
