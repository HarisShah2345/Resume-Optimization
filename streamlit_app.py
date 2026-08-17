"""Streamlit Cloud zero-config entry point.

This shim lives at the repo root so Streamlit Community Cloud finds it
automatically. The actual app lives in `app/app.py`.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` package imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Run the real app. Streamlit executes this module as the entry point.
exec((Path(__file__).parent / "app" / "app.py").read_text(encoding="utf-8"))