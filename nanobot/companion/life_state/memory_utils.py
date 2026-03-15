"""Shared helpers for life-memory indexing and retrieval."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def now_local() -> datetime:
    """Return local aware datetime without microseconds."""
    return datetime.now().astimezone().replace(microsecond=0)


def to_iso(value: datetime) -> str:
    """Serialize datetime in local timezone without microseconds."""
    return value.astimezone().replace(microsecond=0).isoformat()


def parse_iso(value: Any) -> datetime | None:
    """Parse ISO datetime while tolerating naive strings."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now_local().tzinfo)
    return parsed.astimezone().replace(microsecond=0)


def tokenize(text: str) -> list[str]:
    """Tokenize a short text for lexical relevance and clustering."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def compact_text(text: str) -> str:
    """Normalize internal spaces."""
    return re.sub(r"\s+", " ", text or "").strip()

