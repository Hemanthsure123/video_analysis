"""
timestamps.py — Gatekeeper test fixture (isolated; not used by the pipeline).

Parse a human-entered timestamp ("MM:SS" or "H:MM:SS") into whole seconds.
It is the inverse of pipeline.signals.fmt_ts, so a reviewer can jump to an
evidence-log entry by typing the timestamp shown in the report.
"""

from __future__ import annotations

import re

# "H:MM:SS" (minutes must be two digits once hours are present) or "M:SS"/"MM:SS".
# re.ASCII keeps \d from matching non-ASCII digits such as "٣".
_TS_RE = re.compile(r"(?:(\d+):(\d{2})|(\d{1,2})):(\d{2})", re.ASCII)


def parse_ts(text: str) -> int:
    """Return the number of seconds in a "MM:SS" or "H:MM:SS" timestamp.

    Surrounding whitespace is ignored. Raises TypeError for non-strings and
    ValueError for anything that is not a well-formed timestamp, including
    minutes or seconds of 60 or more.
    """
    if not isinstance(text, str):
        raise TypeError(f"timestamp must be a str, got {type(text).__name__}")
    match = _TS_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"invalid timestamp: {text!r}")

    hours, long_minutes, short_minutes, seconds = match.groups()
    h = int(hours) if hours else 0
    m = int(long_minutes or short_minutes)
    s = int(seconds)
    if m >= 60 or s >= 60:
        raise ValueError(f"invalid timestamp: {text!r}")
    return h * 3600 + m * 60 + s
