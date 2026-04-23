from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional


ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "archive"


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    return cleaned[:60] or "domain"


def archive_variant(
    domain: str,
    iteration_count: int,
    stage: str,
    payload: Dict[str, Any],
    score: Optional[float] = None,
) -> str:
    """Persist a generated variant/execution artifact with metadata for traceability."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = _safe_slug(domain)
    filename = f"{timestamp}_iter{iteration_count:03d}_{stage}_{slug}.json"
    path = ARCHIVE_DIR / filename

    record: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "domain": domain,
        "iteration_count": iteration_count,
        "stage": stage,
        "score": score,
        "payload": payload,
    }

    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
