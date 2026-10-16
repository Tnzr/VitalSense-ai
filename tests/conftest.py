import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "ml")):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = os.environ.get("ASCULTO_DATA_DIR", os.path.join(REPO, "data", "HLS-CMDS"))


@pytest.fixture(scope="session")
def data_dir() -> str:
    if not os.path.isfile(os.path.join(DATA_DIR, "HS.csv")):
        pytest.skip(f"HLS-CMDS dataset not found at {DATA_DIR} (run ./setup.sh)")
    return DATA_DIR
