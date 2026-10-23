"""Append-only JSONL audit log. Every guarded tool call leaves one line you can review, label and replay."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


class AuditLog:
    """Writes one JSON object per decision.

    ``redact`` lists argument names whose values are replaced by ``"***"``.
    Set ``log_context=True`` to also store the last user message (off by default: it may hold personal data).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        sink: Callable[[dict[str, Any]], None] | None = None,
        redact: tuple[str, ...] | list[str] = (),
        log_context: bool = False,
    ) -> None:
        self.path = Path(path) if path else None
        self.sink = sink
        self.redact = set(redact)
        self.log_context = log_context
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = dict(entry)
        entry.setdefault("id", uuid.uuid4().hex[:12])
        entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z")
        args = entry.get("arguments")
        if isinstance(args, dict) and self.redact:
            entry["arguments"] = {k: ("***" if k in self.redact else v) for k, v in args.items()}
        if not self.log_context:
            entry.pop("context", None)
        line = json.dumps(entry, sort_keys=True, default=str)
        if self.path:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.sink:
            self.sink(entry)
        return entry


def read_log(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: not valid JSON ({exc})") from exc
    return rows
