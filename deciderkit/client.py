"""Tiny HTTP client for the Strands Decider server (POST /v1/systemone).

Wire format verified against the ``strands-decider`` 0.1.0 package source:
request ``{"state", "questions", "model"}``, response ``{"model", "answers", "usage", "latency_ms"}``.
Stdlib only, so the guard adds no heavy dependencies to your agent.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "http://localhost:8099"
DEFAULT_MODEL = "strands-decider-latest"


class DeciderError(RuntimeError):
    """The Decider server returned an error or an unusable answer."""


class DeciderUnavailable(DeciderError):
    """The server is unreachable, or the circuit breaker is open."""


@dataclass
class DeciderResult:
    answers: dict[str, dict[str, Any]]
    client_latency_ms: float
    server_latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class DeciderClient:
    """Calls the Decider server. A circuit breaker stops a dead server from slowing every tool call."""

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: float = 5.0,
        model: str = DEFAULT_MODEL,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        clock=time.monotonic,
    ) -> None:
        base = (url or os.environ.get("STRANDS_DECIDER_URL") or DEFAULT_URL).rstrip("/")
        if base.endswith("/v1/systemone"):
            base = base[: -len("/v1/systemone")]
        self.url = base
        self.timeout = timeout
        self.model = model
        self._threshold = breaker_threshold
        self._cooldown = breaker_cooldown
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    # -- circuit breaker -------------------------------------------------
    @property
    def breaker_open(self) -> bool:
        with self._lock:
            return self._clock() < self._open_until

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open_until = self._clock() + self._cooldown

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    # -- API ---------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise DeciderUnavailable(f"health check failed: {exc}") from exc

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> DeciderResult:
        if self.breaker_open:
            raise DeciderUnavailable("circuit breaker open: Decider server recently failed")
        body = json.dumps({"state": state, "questions": questions, "model": self.model}).encode()
        req = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            # 4xx means our request was bad (a recipe bug), not that the server is down.
            if 400 <= exc.code < 500:
                raise DeciderError(f"Decider rejected the request ({exc.code}): {detail}") from exc
            self._record_failure()
            raise DeciderUnavailable(f"Decider server error ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self._record_failure()
            raise DeciderUnavailable(f"cannot reach Decider at {self.url}: {exc}") from exc
        except ValueError as exc:
            self._record_failure()
            raise DeciderError(f"Decider returned invalid JSON: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        answers = payload.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            self._record_failure()
            raise DeciderError("Decider response is missing answers for some questions")
        self._record_success()
        return DeciderResult(
            answers=answers,
            client_latency_ms=round(latency, 2),
            server_latency_ms=payload.get("latency_ms"),
            raw=payload,
        )
