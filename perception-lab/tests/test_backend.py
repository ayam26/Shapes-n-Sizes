import json
import socket
import struct
import time
import zlib

import pytest

from app import evaluate, jobs, netguard
from app.png import SIG, solid_png, strip_and_validate
from app.providers import OllamaProvider, parse_output

LOCK = {"endpoint": "http://127.0.0.1:11434/api/chat", "model_id": "m", "options": {"temperature": 0},
        "timeout_s": 1, "think": False}


# ---------- 1. parser ----------
@pytest.mark.parametrize("label", ["physical_vehicle", "flat_depiction", "other", "uncertain"])
def test_parser_accepts_allowed_labels(label):
    status, parsed, err = parse_output(json.dumps({"label": label, "visible_evidence": "x"}))
    assert status == "ok" and parsed["label"] == label and err is None


@pytest.mark.parametrize("raw", [
    '{"label":"truck","visible_evidence":"x"}',          # unknown label
    '{"label":"other"}',                                  # missing key
    '{"label":"other","visible_evidence":"x","conf":1}',  # extra key
    '{"label":"other", "visible_evidence": "x"',          # malformed
    '[{"label":"other","visible_evidence":"x"}]',         # not an object
    'The target is a truck.',                             # prose
    '{"label":"other","visible_evidence":"x"} trailing',  # trailing text
    '',
])
def test_parser_rejects(raw):
    assert parse_output(raw)[0] == "invalid_output"


def test_parser_accepts_single_code_fence():
    assert parse_output('```json\n{"label":"other","visible_evidence":"board"}\n```')[0] == "ok"


def test_provider_timeout(monkeypatch):
    def boom(*a, **k):
        raise socket.timeout("timed out")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    res = OllamaProvider(LOCK, {}).infer(solid_png(4, 4, (0, 0, 0)), "p")
    assert res["status"] == "timeout" and res["raw"] is None


def test_provider_refuses_non_loopback_endpoint():
    with pytest.raises(ValueError):
        OllamaProvider({**LOCK, "endpoint": "http://example.com/api/chat"}, {})


# ---------- 3. ground truth never enters the request ----------
def test_request_contains_only_prompt_and_image():
    img = solid_png(4, 4, (1, 2, 3))
    req = OllamaProvider(LOCK, {"type": "object"}).build_request(img, "PROMPT")
    assert set(req) == {"model", "stream", "think", "keep_alive", "format", "options", "messages"}
    assert len(req["messages"]) == 1 and req["messages"][0]["role"] == "user"
    assert set(req["messages"][0]) == {"role", "content", "images"}
    assert req["messages"][0]["content"] == "PROMPT"


def test_png_metadata_is_stripped():
    png = solid_png(4, 4, (9, 9, 9))
    body = b"Title\x00physical_vehicle truth"
    text = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    tagged = png[:33] + text + png[33:]  # after signature + IHDR
    clean, w, h = strip_and_validate(tagged, 2048)
    assert b"tEXt" not in clean and b"physical_vehicle" not in clean and (w, h) == (4, 4)
    assert clean == png


# ---------- 2. evaluator ----------
def rec(case, method, truth, status, label, sha="h", origin="live", ms=100.0, split="development", obs="clear"):
    return {"case_id": case, "method_id": method, "truth": truth, "status": status, "label": label,
            "image_sha256": sha, "origin": origin, "elapsed_ms": ms, "split": split, "observability": obs}


B, V = evaluate.BASELINE, evaluate.VARIANT


def test_outcomes():
    o = evaluate.outcome
    assert o("flat_depiction", "ok", "physical_vehicle") == "depiction_as_vehicle"
    assert o("physical_vehicle", "ok", "flat_depiction") == "vehicle_as_depiction"
    assert o("other", "ok", "flat_depiction") == "wrong_other_label"
    assert o("physical_vehicle", "ok", "uncertain") == "abstention"
    assert o("other", "ok", "other") == "correct"
    assert o("other", "timeout", None) == "timeout"
    assert o("other", "invalid_output", None) == "invalid_output"
    assert o("other", "queued", None) is None


def test_summary_denominators():
    rs = [rec("a", B, "physical_vehicle", "ok", "physical_vehicle", ms=100),
          rec("b", B, "flat_depiction", "ok", "physical_vehicle", ms=200),
          rec("c", B, "flat_depiction", "ok", "uncertain", ms=300),
          rec("d", B, "other", "invalid_output", None, ms=400),
          rec("e", B, "other", "timeout", None, ms=None),
          rec("f", B, "other", "ok", "other", origin="cached", ms=5)]
    s = evaluate.summarize(rs)
    assert s["attempted"] == 6 and s["valid_responses"] == 4 and s["operational_failures"] == 2
    assert s["abstentions"] == 1
    assert s["correct_over_attempted"] == {"n": 2, "d": 6, "pct": 33.3}
    assert s["accuracy_answered"] == {"n": 2, "d": 3, "pct": 66.7}
    assert s["coverage"] == {"n": 3, "d": 6, "pct": 50.0}
    assert s["false_vehicle_on_depictions"] == {"n": 1, "d": 2, "pct": 50.0}
    assert s["vehicle_recall"] == {"n": 1, "d": 1, "pct": 100.0}
    assert s["control_correct"] == {"n": 1, "d": 3, "pct": 33.3}
    # latency: live ok/invalid only (cached and timeout excluded)
    assert s["latency_ms"]["n"] == 4 and s["latency_ms"]["p50"] == 200 and s["latency_ms"]["p95"] == 400


def test_paired_counts_improvements_and_regressions():
    rs = [rec("1", B, "flat_depiction", "ok", "flat_depiction"), rec("1", V, "flat_depiction", "ok", "flat_depiction"),
          rec("2", B, "flat_depiction", "ok", "physical_vehicle"), rec("2", V, "flat_depiction", "ok", "flat_depiction"),
          rec("3", B, "physical_vehicle", "ok", "physical_vehicle"), rec("3", V, "physical_vehicle", "ok", "uncertain"),
          rec("4", B, "other", "ok", "flat_depiction"), rec("4", V, "other", "ok", "physical_vehicle"),
          rec("5", B, "other", "ok", "other"), rec("5", V, "other", "timeout", None, sha="different")]
    p = evaluate.paired(rs)
    assert (p["both_correct"], p["variant_only"], p["baseline_only"], p["neither"], p["unresolved"]) == (1, 1, 1, 1, 1)
    assert p["pairs"] == 5 and p["hash_mismatch"] == 1


def test_percentile_nearest_rank():
    assert evaluate.percentile([1, 2, 3, 4], 50) == 2
    assert evaluate.percentile(list(range(1, 101)), 95) == 95
    assert evaluate.percentile([], 50) is None


# ---------- API flow, 4/5/6 ----------
def upload(client, case_id, split="development", truth="flat_depiction", png=None):
    meta = {"case_id": case_id, "split": split, "family": "flat_depiction", "truth": truth, "scene_seed": 1,
            "scene_spec": {"conditions": {"target": "board"}}, "renderer_version": "test"}
    return client.post("/api/cases", data={"meta": json.dumps(meta)},
                       files={"image": ("x.png", png or solid_png(8, 8, (10, 20, 30)), "image/png")})


def wait_run(client, run_id, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("completed", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_health_reports_fixture_mode(client):
    h = client.get("/api/health").json()
    assert h["fixture_mode"] is True and h["provider_mode"] == "fixture"
    assert "token" not in json.dumps(h).lower()


def test_upload_validation(client):
    r = client.post("/api/cases", data={"meta": json.dumps({"split": "development", "family": "x", "truth": "other",
                                                            "scene_spec": {}, "renderer_version": "t"})},
                    files={"image": ("x.png", b"not a png", "image/png")})
    assert r.status_code == 422
    big = SIG + b"\0" * (8 * 1024 * 1024 + 10)
    r = upload(client, "big-one", png=big)
    assert r.status_code == 413
    assert upload(client, "../etc", png=solid_png(2, 2, (0, 0, 0))).status_code == 422
    assert upload(client, "c-bad", truth="truck").status_code == 422


def test_case_immutable_and_idempotent(client):
    assert upload(client, "c-imm").json()["existing"] is False
    assert upload(client, "c-imm").json()["existing"] is True
    assert upload(client, "c-imm", png=solid_png(8, 8, (99, 0, 0))).status_code == 409


def test_run_flow_paired_inputs_identical(client):
    upload(client, "c-flow")
    r = client.post("/api/runs", json={"case_ids": ["c-flow"], "method_ids": [B, V], "run_key": "flow-1"}).json()
    run = wait_run(client, r["run_id"])
    assert run["provider_mode"] == "fixture" and run["total"] == 2
    hashes = {x["image_sha256"] for x in run["results"]}
    assert len(hashes) == 1  # byte-identical RGB inputs for both methods
    assert all(x["origin"] == "fixture" for x in run["results"])
    # restart-safe: same run_key returns the same run without new jobs
    again = client.post("/api/runs", json={"case_ids": ["c-flow"], "method_ids": [B, V], "run_key": "flow-1"}).json()
    assert again == {"run_id": r["run_id"], "existing": True}
    m = client.get(f"/api/metrics?run_id={r['run_id']}").json()
    assert m["paired_by_split"]["development"]["pairs"] == 1
    for fmt in ("csv", "jsonl", "md", "zip"):
        assert client.get(f"/api/export/{r['run_id']}?fmt={fmt}").status_code == 200


def test_heldout_sealed_until_frozen(client, tmp_root):
    upload(client, "c-held", split="heldout")
    r = client.post("/api/runs", json={"case_ids": ["c-held"], "method_ids": [B]})
    assert r.status_code == 423
    files = [tmp_root / "experiments" / n for n in ("prompts.json", "model-lock.json")]
    originals = [f.read_text() for f in files]
    try:
        for f in files:
            d = json.loads(f.read_text())
            d["frozen"] = True
            f.write_text(json.dumps(d))
        r = client.post("/api/runs", json={"case_ids": ["c-held"], "method_ids": [B]})
        assert r.status_code == 200
        wait_run(client, r.json()["run_id"])
    finally:
        for f, o in zip(files, originals):
            f.write_text(o)


def test_cache_hits_and_invalidates_on_prompt_change(client, tmp_root, monkeypatch):
    calls = []

    class FakeLive:
        mode = "live"

        def __init__(self, lock, schema):
            pass

        def infer(self, image, prompt, think=None):
            calls.append(prompt)
            return {"status": "completed", "raw": '{"label":"other","visible_evidence":"x"}', "elapsed_ms": 1.0, "meta": {}}

    monkeypatch.setattr(jobs, "make_provider", FakeLive)
    upload(client, "c-cache")
    body = {"case_ids": ["c-cache"], "method_ids": [B], "use_cache": True}
    run1 = wait_run(client, client.post("/api/runs", json=body).json()["run_id"])
    run2 = wait_run(client, client.post("/api/runs", json=body).json()["run_id"])
    assert run1["results"][0]["origin"] == "live" and run2["results"][0]["origin"] == "cached" and len(calls) == 1
    p = tmp_root / "experiments" / "prompts.json"
    original = p.read_text()
    try:
        d = json.loads(original)
        d["methods"][B]["prompt"] += " Changed."
        p.write_text(json.dumps(d))
        run3 = wait_run(client, client.post("/api/runs", json=body).json()["run_id"])
        assert run3["results"][0]["origin"] == "live" and len(calls) == 2
    finally:
        p.write_text(original)


def test_failure_state_is_recorded(client, monkeypatch):
    class FakeTimeout:
        mode = "live"

        def __init__(self, lock, schema):
            pass

        def infer(self, image, prompt, think=None):
            return {"status": "timeout", "raw": None, "elapsed_ms": 1000.0, "meta": {}}

    monkeypatch.setattr(jobs, "make_provider", FakeTimeout)
    upload(client, "c-fail")
    run = wait_run(client, client.post("/api/runs", json={"case_ids": ["c-fail"], "method_ids": [B]}).json()["run_id"])
    assert run["results"][0]["status"] == "timeout" and run["error_summary"][0]["status"] == "timeout"


def test_reconcile_requeues_running(client):
    from app import db
    upload(client, "c-rec")
    rid = client.post("/api/runs", json={"case_ids": ["c-rec"], "method_ids": [B]}).json()["run_id"]
    wait_run(client, rid)
    db.execute("UPDATE results SET status='running' WHERE run_id=?", (rid,))
    assert jobs.reconcile_after_restart() == 1
    assert db.one("SELECT status FROM results WHERE run_id=?", (rid,))["status"] in ("queued", "running", "ok")


# ---------- 7. offline guard ----------
def test_offline_guard_blocks_external_allows_loopback():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    netguard.enable()
    try:
        s = socket.socket()
        with pytest.raises(ConnectionRefusedError, match="blocked"):
            s.connect(("192.0.2.1", 80))
        s.close()
        c = socket.socket()
        c.connect(srv.getsockname())  # loopback still works
        c.close()
    finally:
        netguard.disable()
        srv.close()


def test_run_rejects_bad_model_lock(client):
    upload(client, "c-lock")
    for bad in ("../state/run.json", "prompts.json", "model-lock-missing.json"):
        r = client.post("/api/runs", json={"case_ids": ["c-lock"], "method_ids": [B], "model_lock": bad})
        assert r.status_code == 422


def test_heldout2_sealed_until_interventions_frozen(client, tmp_root):
    upload(client, "c-h2", split="heldout2")
    iv = tmp_root / "experiments" / "interventions.json"
    iv.write_text(json.dumps({"frozen": False, "methods": {"rgb_think": {"prompt": "P", "think": True}}}))
    files = [tmp_root / "experiments" / n for n in ("prompts.json", "model-lock.json")]
    originals = [f.read_text() for f in files]
    try:
        for f in files:
            d = json.loads(f.read_text()); d["frozen"] = True; f.write_text(json.dumps(d))
        assert client.post("/api/runs", json={"case_ids": ["c-h2"], "method_ids": [B]}).status_code == 423
        iv.write_text(json.dumps({"frozen": True, "methods": {"rgb_think": {"prompt": "P", "think": True}}}))
        r = client.post("/api/runs", json={"case_ids": ["c-h2"], "method_ids": [B, "rgb_think"]})
        assert r.status_code == 200
        run = wait_run(client, r.json()["run_id"])
        assert run["model_config"]["methods"]["rgb_think"]["think"] is True
        assert run["model_config"]["methods"]["rgb_think"]["source"] == "interventions.json"
    finally:
        for f, o in zip(files, originals):
            f.write_text(o)
        iv.unlink()


def test_think_flag_reaches_request():
    req = OllamaProvider(LOCK, {}).build_request(solid_png(2, 2, (0, 0, 0)), "P", think=True)
    assert req["think"] is True
    assert OllamaProvider(LOCK, {}).build_request(solid_png(2, 2, (0, 0, 0)), "P")["think"] is False
