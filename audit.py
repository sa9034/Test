"""Audit trail — saves structured JSON logs for each council query."""

from __future__ import annotations

import json
import time
from pathlib import Path

import config
from models import AuditLogEntry


def save_audit_log(entry: AuditLogEntry) -> Path:
    """Save an audit log entry as a JSON file and return the file path."""
    config.AUDIT_LOG_DIR.mkdir(exist_ok=True)
    filename = f"audit_{time.strftime('%Y%m%d_%H%M%S')}_{id(entry) % 10000:04d}.json"
    filepath = config.AUDIT_LOG_DIR / filename
    filepath.write_text(
        json.dumps(entry.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    return filepath
