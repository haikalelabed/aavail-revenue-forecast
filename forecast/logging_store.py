"""Store prediction events in SQLite for concurrent API requests."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class PredictionLog:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, prediction, latency_ms):
        event = dict(prediction, logged_at=datetime.now(timezone.utc).isoformat(),
                     latency_ms=round(latency_ms, 3))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, event TEXT)")
            conn.execute("INSERT INTO predictions (event) VALUES (?)", (json.dumps(event),))

    def read(self, limit=None):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
            if not exists:
                return []
            if limit is None:
                rows = conn.execute("SELECT event FROM predictions ORDER BY id DESC")
            else:
                rows = conn.execute("SELECT event FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            return [json.loads(row[0]) for row in rows]
