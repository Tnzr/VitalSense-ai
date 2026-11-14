"""User accounts, sessions and saved auscultation exams (SQLite, stdlib only).

The schema is deliberately plain SQL with JSON payloads so it can move to
Postgres (Neon) without changes to the callers.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION_COOKIE = "asculto_session"
SESSION_TTL_S = int(os.environ.get("ASCULTO_SESSION_TTL_S", str(14 * 24 * 3600)))
MIN_PASSWORD = 8
_SCRYPT = (2 ** 14, 8, 1)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  pw_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS exams (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  site TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  filename TEXT NOT NULL DEFAULT '',
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS exams_user_time ON exams(user_id, recorded_at);
"""


def db_path() -> str:
    return os.environ.get("ASCULTO_DB", os.path.join(REPO, "var", "asculto.db"))


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    n, r, p = _SCRYPT
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(bytes.fromhex(dk)))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), dk)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def create_user(email: str, password: str) -> int:
    email = normalize_email(email)
    if "@" not in email or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    with _conn() as con:
        try:
            cur = con.execute("INSERT INTO users(email, pw_hash, created_at) VALUES (?, ?, ?)",
                              (email, hash_password(password), _now_iso()))
        except sqlite3.IntegrityError:
            raise ValueError("An account with this email already exists.") from None
        return int(cur.lastrowid)


def authenticate(email: str, password: str) -> Optional[int]:
    with _conn() as con:
        row = con.execute("SELECT id, pw_hash FROM users WHERE email = ?", (normalize_email(email),)).fetchone()
    if row is None:
        verify_password(password or "", hash_password("timing-equaliser"))
        return None
    return int(row["id"]) if verify_password(password or "", row["pw_hash"]) else None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
        con.execute("INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                    (_token_hash(token), user_id, int(time.time()) + SESSION_TTL_S))
    return token


def user_for_token(token: Optional[str]) -> Optional[Dict]:
    if not token:
        return None
    with _conn() as con:
        row = con.execute(
            "SELECT u.id, u.email, u.created_at FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at >= ?", (_token_hash(token), int(time.time()))).fetchone()
    return dict(row) if row else None


def end_session(token: Optional[str]) -> None:
    if token:
        with _conn() as con:
            con.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def delete_user(user_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))


def add_exam(user_id: int, recorded_at: str, site: str, note: str, filename: str, data: Dict) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO exams(user_id, created_at, recorded_at, site, note, filename, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, _now_iso(), recorded_at, site, note, filename, json.dumps(data)))
        return int(cur.lastrowid)


def _exam(row: sqlite3.Row) -> Dict:
    out = {k: row[k] for k in ("id", "created_at", "recorded_at", "site", "note", "filename")}
    out.update(json.loads(row["data"]))
    return out


def list_exams(user_id: int) -> List[Dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM exams WHERE user_id = ? ORDER BY recorded_at, id", (user_id,)).fetchall()
    return [_exam(r) for r in rows]


def delete_exam(user_id: int, exam_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM exams WHERE id = ? AND user_id = ?", (exam_id, user_id))
        return cur.rowcount > 0
