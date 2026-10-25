"""
report_index.py — Gatekeeper test fixture (isolated; not used by the pipeline).

A small SQLite index of finished audit reports, so reviewers can find past
reports by candidate name from a search box and sort the results.
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY,
    candidate   TEXT NOT NULL,
    recorded_on TEXT NOT NULL,
    verdict     TEXT NOT NULL
)
"""


def open_index(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def add_report(conn: sqlite3.Connection, candidate: str, recorded_on: str, verdict: str) -> int:
    cur = conn.execute(
        "INSERT INTO reports (candidate, recorded_on, verdict) VALUES (?, ?, ?)",
        (candidate, recorded_on, verdict),
    )
    conn.commit()
    return cur.lastrowid


def search_reports(conn: sqlite3.Connection, name_query: str, sort_by: str = "recorded_on") -> list[dict]:
    """Return reports whose candidate name contains name_query, newest first.

    name_query comes straight from the search box; sort_by from the column the
    user clicked in the results table.
    """
    sql = (
        "SELECT id, candidate, recorded_on, verdict FROM reports "
        f"WHERE candidate LIKE '%{name_query}%' "
        f"ORDER BY {sort_by} DESC"
    )
    return [dict(row) for row in conn.execute(sql)]
