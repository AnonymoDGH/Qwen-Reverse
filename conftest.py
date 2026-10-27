"""Pytest bootstrap: make the repository root importable.

Keeps the test suite hermetic (no absolute paths, works on CI).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
