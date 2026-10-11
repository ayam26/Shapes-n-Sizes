import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="pl-test-"))
shutil.copytree(PROJECT / "experiments", TMP / "experiments")
# Tests start from an UNFROZEN copy so the held-out seal can be exercised regardless of repo state.
for _name in ("prompts.json", "model-lock.json"):
    _p = TMP / "experiments" / _name
    _d = json.loads(_p.read_text())
    _d["frozen"] = False
    _p.write_text(json.dumps(_d))
os.environ["PL_ROOT"] = str(TMP)
os.environ["PL_DATA"] = str(TMP / "data")
os.environ["PL_PROVIDER"] = "fixture"
sys.path.insert(0, str(PROJECT / "backend"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_root():
    return TMP
