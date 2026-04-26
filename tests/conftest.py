"""Pytest configuration for claude_control tests."""

import sys
from pathlib import Path

# Make ``claude_control`` importable when running pytest from the repo root
# without requiring an editable install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
