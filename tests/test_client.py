import pytest

from deciderkit import DeciderClient, DeciderError, DeciderUnavailable


def test_roundtrip_and_health(server, client):
    assert client.health()["status"] == "ok"
    res = client.ask({"conversation": [], "proposed_tool_call": {"tool": "t", "arguments": {}}},
                     {"q": {"type": "noul", "instructions": "ok?"}})
    assert 0 <= res.answers["q"]["noul"] <= 1 and res.client_latency_ms >= 0


def test_unreachable_raises_unavailable():
    with pytest.raises(DeciderUnavailable):
        DeciderClient("http://127.0.0.1:9", timeout=0.5).ask("s", {"q": {"type": "noul", "instructions": "x"}})


def test_bad_request_is_not_an_outage(server):
    c = DeciderClient(server.url, breaker_threshold=1)
    with pytest.raises(DeciderError) as e:
        c.ask("s", {"q": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}})
    assert not isinstance(e.value, DeciderUnavailable) and not c.breaker_open


def test_url_forms(monkeypatch):
    assert DeciderClient("http://h:1/v1/systemone/").url == "http://h:1"
    monkeypatch.setenv("STRANDS_DECIDER_URL", "http://env:9")
    assert DeciderClient().url == "http://env:9"
