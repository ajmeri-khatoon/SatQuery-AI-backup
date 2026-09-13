"""Expose the teammate app package as ``app`` for its existing test contract."""

import sys
from pathlib import Path

APP_PARENT = str(Path(__file__).resolve().parents[1])
if APP_PARENT not in sys.path:
    sys.path.insert(0, APP_PARENT)