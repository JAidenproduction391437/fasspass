"""Tiny storage layer: Postgres when DATABASE_URL is set (Render), SQLite otherwise (local dev)."""
import json
import os
import sqlite3
import threading
import time

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "facepass.db"))

_lock = threading.Lock()

if DATABASE_URL:
    import psycopg

    PH = "%s"

    def _connect():
        return psycopg.connect(DATABASE_URL, autocommit=True)

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            descriptors TEXT NOT NULL,
            photo BYTEA,
            created_at DOUBLE PRECISION NOT NULL
        )
    """
else:
    PH = "?"

    def _connect():
        conn = sqlite3.connect(SQLITE_PATH)
        conn.isolation_level = None  # autocommit
        return conn

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            descriptors TEXT NOT NULL,
            photo BLOB,
            created_at REAL NOT NULL
        )
    """


def _run(sql, params=(), fetch=False):
    sql = sql.replace("?", PH)
    with _lock:
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall() if fetch else None
        finally:
            conn.close()


def init():
    _run(SCHEMA)


# In-memory cache of (id, name, descriptors) so verification doesn't hit the DB every scan.
_cache = None
_cache_time = 0.0
CACHE_SECONDS = 30


def _invalidate():
    global _cache
    _cache = None


def add_person(pid, name, descriptors, photo_bytes):
    _run(
        "INSERT INTO people (id, name, descriptors, photo, created_at) VALUES (?, ?, ?, ?, ?)",
        (pid, name, json.dumps(descriptors), photo_bytes, time.time()),
    )
    _invalidate()


def delete_person(pid):
    _run("DELETE FROM people WHERE id = ?", (pid,))
    _invalidate()


def list_people():
    rows = _run("SELECT id, name, created_at FROM people ORDER BY created_at DESC", fetch=True)
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def all_descriptors():
    """Returns [(id, name, [[128 floats], ...]), ...] (cached briefly)."""
    global _cache, _cache_time
    if _cache is None or time.time() - _cache_time > CACHE_SECONDS:
        rows = _run("SELECT id, name, descriptors FROM people", fetch=True)
        _cache = [(r[0], r[1], json.loads(r[2])) for r in rows]
        _cache_time = time.time()
    return _cache


def get_photo(pid):
    rows = _run("SELECT photo FROM people WHERE id = ?", (pid,), fetch=True)
    if not rows or rows[0][0] is None:
        return None
    return bytes(rows[0][0])
