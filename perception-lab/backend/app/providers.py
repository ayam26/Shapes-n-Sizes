"""Local inference providers and the strict output parser.

A provider sees only image bytes and a prompt string. Case metadata (truth, family,
filenames, scene spec) is never passed in, so it cannot reach the model."""
import base64
import json
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import config

FENCE = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.S)


def parse_output(raw: str):
    """Return (status, parsed, error). No semantic repair: one JSON object, optionally
    inside a single code fence, with exactly the keys label and visible_evidence."""
    text = (raw or "").strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return "invalid_output", None, f"json: {e}"
    if not isinstance(obj, dict):
        return "invalid_output", None, "not an object"
    if set(obj) != {"label", "visible_evidence"}:
        return "invalid_output", None, f"keys {sorted(obj)}"
    if obj["label"] not in config.LABELS:
        return "invalid_output", None, f"unknown label {obj['label']!r}"
    if not isinstance(obj["visible_evidence"], str) or len(obj["visible_evidence"]) > 2000:
        return "invalid_output", None, "visible_evidence must be a string <= 2000 chars"
    return "ok", obj, None


class OllamaProvider:
    mode = "live"

    def __init__(self, lock: dict, output_schema: dict):
        self.lock = lock
        self.output_schema = output_schema
        host = urlparse(lock["endpoint"]).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("provider endpoint must be loopback (local-only requirement)")

    def build_request(self, image_png: bytes, prompt: str, think=None) -> dict:
        return {
            "model": self.lock["model_id"],
            "stream": False,
            "think": self.lock.get("think", False) if think is None else think,
            "keep_alive": "30m",
            "format": self.output_schema,
            "options": self.lock["options"],
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(image_png).decode()]}],
        }

    def infer(self, image_png: bytes, prompt: str, think=None) -> dict:
        body = json.dumps(self.build_request(image_png, prompt, think)).encode()
        req = urllib.request.Request(self.lock["endpoint"], data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.lock["timeout_s"]) as r:
                resp = json.load(r)
        except (socket.timeout, TimeoutError):
            return {"status": "timeout", "raw": None, "elapsed_ms": (time.perf_counter() - t0) * 1000, "meta": {}}
        except urllib.error.URLError as e:
            status = "timeout" if isinstance(e.reason, (socket.timeout, TimeoutError)) else "provider_error"
            return {"status": status, "raw": None, "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    "meta": {"error": str(e)[:500]}}
        elapsed = (time.perf_counter() - t0) * 1000
        meta = {k: resp.get(k) for k in ("model", "total_duration", "load_duration",
                                          "prompt_eval_count", "eval_count", "done_reason")}
        if resp.get("message", {}).get("thinking"):
            meta["thinking"] = resp["message"]["thinking"][:4000]  # stored as stated reasoning, never parsed
        return {"status": "completed", "raw": resp.get("message", {}).get("content", ""),
                "elapsed_ms": elapsed, "meta": meta}

    def tags(self) -> list:
        url = self.lock["endpoint"].replace("/api/chat", "/api/tags")
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r).get("models", [])


class FixtureProvider:
    """Development fixture. Never eligible as evidence of local model performance."""
    mode = "fixture"

    def __init__(self, lock: dict, output_schema: dict):
        self.lock = lock

    def infer(self, image_png: bytes, prompt: str, think=None) -> dict:
        time.sleep(0.05)
        return {"status": "completed", "elapsed_ms": 50.0, "meta": {"fixture": True},
                "raw": json.dumps({"label": "uncertain", "visible_evidence": "FIXTURE OUTPUT - not a model response"})}

    def tags(self) -> list:
        return []


def make_provider(lock: dict, output_schema: dict):
    cls = FixtureProvider if config.PROVIDER == "fixture" else OllamaProvider
    return cls(lock, output_schema)
