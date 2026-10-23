"""A fake Decider server that speaks the real wire format (POST /v1/systemone, GET /health).

It uses a simple text-matching rule instead of a model, so you can build and test with no GPU:
the share of argument values found in the user/tool-result text becomes P(true).
All values found -> 0.95, none -> 0.05, some -> in between (lands in the "uncertain" band).

WARNING: its answers are NOT Decider's answers. Never report numbers measured against it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _flatten(values: Any) -> list[str]:
    if isinstance(values, dict):
        return [s for v in values.values() for s in _flatten(v)]
    if isinstance(values, (list, tuple)):
        return [s for v in values for s in _flatten(v)]
    if values is None or isinstance(values, bool):
        return []
    if isinstance(values, float) and values.is_integer():
        return [str(int(values)), f"{values:.2f}"]
    return [str(values)]


def grounded_fraction(state: Any) -> float:
    if not isinstance(state, dict):
        return 0.5
    evidence = " ".join(
        m.get("text", "") for m in state.get("conversation", []) if m.get("role") in ("user", "tool_result")
    ).lower()
    values = _flatten(state.get("proposed_tool_call", {}).get("arguments", {}))
    if not values:
        return 1.0
    return sum(1 for v in values if v.lower() in evidence) / len(values)


def answer_question(spec: dict[str, Any], state: Any) -> dict[str, Any]:
    if spec["type"] == "noul":
        frac = grounded_fraction(state)
        return {"type": "noul", "noul": round(0.05 + 0.90 * frac, 4)}
    if spec["type"] == "choice":
        text = json.dumps(state).lower()
        options = list(spec["criteria"])
        hit = next((o for o in options if o.lower() in text), None)
        chosen, conf = (hit, 0.9) if hit else (options[0], 0.4)
        probs = {o: (conf if o == chosen else (1 - conf) / (len(options) - 1)) for o in options}
        return {"type": "choice", "choice": chosen, "probabilities": probs, "confidence": conf}
    raise ValueError(f"mock does not support question type {spec['type']!r}")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass

    def _send(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": "mock-decider", "mock": True})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/systemone":
            return self._send(404, {"detail": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            answers = {n: answer_question(q, body["state"]) for n, q in body["questions"].items()}
        except (ValueError, KeyError) as exc:
            return self._send(422, {"detail": str(exc)})
        mode = self.server.mode  # type: ignore[attr-defined]
        if mode == "error":
            return self._send(500, {"detail": "mock forced error"})
        self._send(200, {"model": "mock-decider", "answers": answers,
                         "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_ms": 1.0})


class MockDeciderServer:
    """Run in a background thread: ``with MockDeciderServer() as srv: client = DeciderClient(srv.url)``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.mode = "ok"  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def set_mode(self, mode: str) -> None:
        self._server.mode = mode  # type: ignore[attr-defined]  # "ok" or "error"

    def start(self) -> "MockDeciderServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "MockDeciderServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve(host: str = "127.0.0.1", port: int = 8099) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.mode = "ok"  # type: ignore[attr-defined]
    print(f"MOCK Decider on http://{host}:{port}  (testing only, not the real model)")
    server.serve_forever()
