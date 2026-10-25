"""
report_endpoint.py — Gatekeeper test fixture (isolated; not used by the pipeline).

Framework-agnostic handler for GET /reports/<job_id>. It returns a finished
audit report as JSON so an external dashboard can poll for results.
"""

from __future__ import annotations

import json
import os
import re
import traceback

REPORT_DIR = os.environ.get(
    "AUDIT_REPORT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
)
_JOB_ID_RE = re.compile(r"[0-9a-f]{12}")


def _load_report(job_id: str) -> dict:
    with open(os.path.join(REPORT_DIR, f"{job_id}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def get_report(request: dict) -> tuple[int, dict]:
    """Handle GET /reports/<job_id>. Returns (http_status, json_body)."""
    try:
        job_id = request["params"]["job_id"]
        if not _JOB_ID_RE.fullmatch(job_id):
            return 400, {"error": "job_id must be 12 lowercase hex characters"}
        report = _load_report(job_id)
        return 200, {"job_id": job_id, "report": report}
    except Exception as exc:
        return 500, {
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        }
