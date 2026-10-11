import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="pl-test-"))
shutil.copytree(PROJECT / "experiments", TMP / "experiments")
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
