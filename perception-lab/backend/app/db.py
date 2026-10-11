import sqlite3
import threading
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  case_id TEXT PRIMARY KEY, schema_version INT, split TEXT, family TEXT, scene_seed INT,
  scene_spec TEXT, conditions TEXT, image_path TEXT, image_sha256 TEXT, width INT, height INT,
  truth TEXT, observability TEXT, source_kind TEXT, license TEXT, renderer_version TEXT,
  render_status TEXT DEFAULT 'valid', render_note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, run_key TEXT UNIQUE, label TEXT, created_at TEXT,
  model_config TEXT, model_config_hash TEXT, method_ids TEXT, status TEXT,
  provider_mode TEXT, use_cache INT, offline_guard INT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS results(
  result_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, seq INT, case_id TEXT, method_id TEXT,
  prompt_hash TEXT, image_sha256 TEXT, model_id TEXT, model_digest TEXT, quantization TEXT,
  runtime_version TEXT, options TEXT, status TEXT, raw_output TEXT, parsed TEXT, label TEXT,
  parse_error TEXT, elapsed_ms REAL, origin TEXT, cache_key TEXT, provider_meta TEXT,
  queued_at TEXT, started_at TEXT, finished_at TEXT,
  UNIQUE(run_id, case_id, method_id));
CREATE TABLE IF NOT EXISTS cache(cache_key TEXT PRIMARY KEY, result_id INT, created_at TEXT);
CREATE TABLE IF NOT EXISTS warmups(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, model_id TEXT, status TEXT,
  elapsed_ms REAL, load_ms REAL);
"""

write_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    config.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)


def rows(sql: str, args=()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def one(sql: str, args=()):
    r = rows(sql, args)
    return r[0] if r else None


def execute(sql: str, args=()) -> int:
    with write_lock, connect() as conn:
        cur = conn.execute(sql, args)
        return cur.lastrowid
