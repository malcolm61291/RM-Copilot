"""Lightweight tracing: every node, tool call and LLM call becomes a span with timing and a
compact summary. Spans are kept in memory for the UI and appended to runs/<thread>.jsonl.

In production this maps 1:1 onto OpenTelemetry spans or LangSmith/Langfuse runs.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

from crm_agent.config import settings

_current_thread: ContextVar[str] = ContextVar("thread_id", default="default")
_SPANS: dict[str, list[dict]] = {}


def set_thread(thread_id: str) -> None:
    _current_thread.set(thread_id)


def current_thread() -> str:
    return _current_thread.get()


def _summarise(value: Any, limit: int = 160) -> Any:
    text = json.dumps(value, default=str, ensure_ascii=False)
    return value if len(text) <= limit else text[:limit] + "…"


@contextmanager
def span(kind: str, name: str, **attrs: Any) -> Iterator[dict]:
    record = {"span_id": uuid.uuid4().hex[:8], "kind": kind, "name": name,
              "ts": datetime.now(timezone.utc).isoformat(),
              "attrs": {k: _summarise(v) for k, v in attrs.items()}}
    start = time.perf_counter()
    try:
        yield record
        record.setdefault("status", "ok")
    except Exception as exc:  # recorded, then re-raised
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["ms"] = round((time.perf_counter() - start) * 1000, 1)
        if "result" in record:
            record["result"] = _summarise(record["result"])
        thread = current_thread()
        _SPANS.setdefault(thread, []).append(record)
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.runs_dir / f"{thread}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def spans(thread_id: str | None = None) -> list[dict]:
    return list(_SPANS.get(thread_id or current_thread(), []))


def clear(thread_id: str | None = None) -> None:
    _SPANS.pop(thread_id or current_thread(), None)
