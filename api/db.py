"""Tiny SQLite store for scan statistics and user feedback.

Privacy by design: we store NO photos, NO IP addresses and NO personal data –
only the predicted label, confidence and optional feedback.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scan_id INTEGER,
    predicted_label TEXT NOT NULL,
    correct INTEGER NOT NULL,
    actual_label TEXT,
    comment TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def log_scan(self, label: str | None, confidence: float | None, status: str) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO scans (created_at, label, confidence, status) VALUES (?, ?, ?, ?)",
                (_now(), label, confidence, status),
            )
            return int(cur.lastrowid)

    def add_feedback(self, scan_id: int | None, predicted_label: str, correct: bool,
                     actual_label: str | None, comment: str | None) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO feedback (created_at, scan_id, predicted_label, correct, actual_label, comment) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), scan_id, predicted_label, int(correct), actual_label, comment),
            )
            return int(cur.lastrowid)

    def stats(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            by_status = dict(conn.execute("SELECT status, COUNT(*) FROM scans GROUP BY status").fetchall())
            top = conn.execute(
                "SELECT label, COUNT(*) AS n FROM scans WHERE label IS NOT NULL "
                "GROUP BY label ORDER BY n DESC LIMIT 10"
            ).fetchall()
            fb_total, fb_correct = conn.execute("SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM feedback").fetchone()
            weak = conn.execute(
                "SELECT predicted_label, COUNT(*) AS n FROM feedback WHERE correct = 0 "
                "GROUP BY predicted_label ORDER BY n DESC LIMIT 5"
            ).fetchall()
        return {
            "total_scans": total,
            "by_status": by_status,
            "top_labels": [{"label": label, "count": n} for label, n in top],
            "feedback_count": fb_total,
            "feedback_accuracy": round(fb_correct / fb_total, 3) if fb_total else None,
            "most_reported_wrong": [{"label": label, "count": n} for label, n in weak],
        }
