"""SQLite persistence: file registry, chunks, chat sessions, eval runs."""
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL, modality TEXT NOT NULL,
    size INTEGER, status TEXT NOT NULL, progress REAL DEFAULT 0, stage TEXT DEFAULT '',
    error TEXT DEFAULT '', n_chunks INTEGER DEFAULT 0, thumb TEXT DEFAULT '',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, file_id TEXT NOT NULL, modality TEXT NOT NULL, kind TEXT NOT NULL,
    content TEXT NOT NULL, page INTEGER, t_start REAL, t_end REAL, thumb TEXT DEFAULT '',
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    citations TEXT DEFAULT '[]', meta TEXT DEFAULT '{}', created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE TABLE IF NOT EXISTS eval_runs (id TEXT PRIMARY KEY, created REAL NOT NULL, summary TEXT, details TEXT);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Database:
    def __init__(self, path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        # Anything mid-ingestion when the process died is marked failed so the user can retry.
        self._conn.execute(
            "UPDATE files SET status='failed', error='interrupted' WHERE status IN ('queued','processing')"
        )
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    # ---- files ----
    def add_file(self, name: str, path: str, modality: str, size: int) -> str:
        fid = new_id()
        with self.tx() as c:
            c.execute(
                "INSERT INTO files(id,name,path,modality,size,status,created) VALUES(?,?,?,?,?,?,?)",
                (fid, name, path, modality, size, "queued", time.time()),
            )
        return fid

    def update_file(self, fid: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ",".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE files SET {cols} WHERE id=?", (*fields.values(), fid))

    def list_files(self) -> list[dict]:
        return self.query("SELECT * FROM files ORDER BY created DESC")

    def get_file(self, fid: str) -> dict | None:
        return self.one("SELECT * FROM files WHERE id=?", (fid,))

    def delete_file(self, fid: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
            c.execute("DELETE FROM files WHERE id=?", (fid,))

    # ---- chunks ----
    def add_chunks(self, chunks: list[dict]) -> None:
        with self.tx() as c:
            c.executemany(
                "INSERT INTO chunks(id,file_id,modality,kind,content,page,t_start,t_end,thumb,meta)"
                " VALUES(:id,:file_id,:modality,:kind,:content,:page,:t_start,:t_end,:thumb,:meta)",
                [{**ch, "meta": json.dumps(ch.get("meta", {}))} for ch in chunks],
            )

    def all_chunks(self) -> list[dict]:
        return self.query(
            "SELECT c.*, f.name AS file_name FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.status='indexed'"
        )

    def get_chunks(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.query(
            f"SELECT c.*, f.name AS file_name FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id IN ({q})",
            tuple(ids),
        )
        return {r["id"]: r for r in rows}

    # ---- sessions ----
    def create_session(self, title: str = "New chat") -> dict:
        sid = new_id()
        with self.tx() as c:
            c.execute("INSERT INTO sessions(id,title,created) VALUES(?,?,?)", (sid, title, time.time()))
        return {"id": sid, "title": title}

    def add_message(self, sid: str, role: str, content: str, citations: list | None = None, meta: dict | None = None) -> str:
        mid = new_id()
        with self.tx() as c:
            c.execute(
                "INSERT INTO messages(id,session_id,role,content,citations,meta,created) VALUES(?,?,?,?,?,?,?)",
                (mid, sid, role, content, json.dumps(citations or []), json.dumps(meta or {}), time.time()),
            )
        return mid

    def messages(self, sid: str) -> list[dict]:
        rows = self.query("SELECT * FROM messages WHERE session_id=? ORDER BY created", (sid,))
        for r in rows:
            r["citations"] = json.loads(r["citations"])
            r["meta"] = json.loads(r["meta"])
        return rows
