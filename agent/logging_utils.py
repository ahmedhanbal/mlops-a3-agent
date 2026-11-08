"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from typing import Any

_SECRET_KEY = re.compile(r"(token|password|passwd|secret|authorization|api[_-]?key|credential)", re.I)
_SECRET_VALUE = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWT
    re.compile(r"(?i)(password|token|secret|api[_-]?key)\s*[=:]\s*\S+"),
]
REDACTED = "***REDACTED***"


def redact(value: Any) -> Any:
    """Recursively mask secrets in dicts/lists/strings."""
    if isinstance(value, dict):
        return {k: (REDACTED if _SECRET_KEY.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pat in _SECRET_VALUE:
            value = pat.sub(REDACTED, value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(redact(payload), default=str)


def get_logger(name: str = "mlops_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    get_logger().log(level, event, extra={"fields": fields})


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
