import json

import pytest

from conftest import convo
from deciderkit import AuditLog, DeciderClient, DecisionEngine, load_recipe

RECIPE = "recipes/support_ops.yaml"
GOOD = convo(("user", "Please refund 42.50 on ORD-482913, it arrived broken. I want a refund."))


def engine(client, recipe=RECIPE, **kw):
    return DecisionEngine(recipe, client=client, **kw)


def test_grounded_call_proceeds(client):
    d = engine(client).evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD)
    assert d.action == "proceed" and d.source == "decider"
    assert all(c.verdict == "pass" for c in d.checks)


def test_guessed_order_id_is_guided(client):
    conv = convo(("user", "refund my order please"))
    d = engine(client).evaluate("refund_order", {"order_id": "ORD-123456", "amount": 20.0}, conv)
    assert d.action == "guide"
    assert "failed" in d.reason and d.feedback


def test_partly_grounded_is_uncertain_then_guided(client):
    conv = convo(("user", "refund ORD-482913 please, I want a refund"))
    d = engine(client).evaluate("refund_order", {"order_id": "ORD-482913", "amount": 99.0}, conv)
    assert d.action == "guide" and d.reason.startswith("uncertain")


def test_tool_results_count_as_evidence(client):
    conv = convo(
        ("tool_result", "orders: ORD-111222 total 30.00"),
        ("user", "refund the one you found, I want a refund"),
    )
    d = engine(client).evaluate("refund_order", {"order_id": "ORD-111222", "amount": 30.0}, conv)
    assert d.action == "proceed"


def test_assistant_text_is_not_evidence(client):
    conv = convo(("assistant", "How about ORD-999999 for 10.00?"), ("user", "refund it"))
    d = engine(client).evaluate("refund_order", {"order_id": "ORD-999999", "amount": 10.0}, conv)
    assert d.action == "guide"


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"order_id": "ORD-482913", "amount": 900}, "over $500"),
        ({"order_id": "ORD-482913", "amount": 0}, "at least 0.01"),
        ({"order_id": "bad-id", "amount": 5}, "invalid format"),
        ({"amount": 5}, "required"),
    ],
)
def test_hard_rules_deny_before_decider(client, args, fragment):
    d = engine(client).evaluate("refund_order", args, GOOD)
    assert d.action == "deny" and d.source == "hard_rule" and fragment in d.reason
    assert d.checks == []  # Decider was never asked


def test_skip_and_unlisted_tools(client):
    e = engine(client)
    assert e.evaluate("lookup_orders", {"email": "x"}, []).action == "proceed"
    assert e.evaluate("mystery_tool", {}, []).source == "unlisted"
    strict = engine(client, {"unlisted_tools": "deny", "tools": {"a": {"skip": True}}})
    assert strict.evaluate("mystery_tool", {}, []).action == "deny"


def test_shadow_mode_logs_but_lets_through(client, tmp_path):
    log = tmp_path / "audit.jsonl"
    r = {"tools": {"refund_order": {"mode": "shadow", "checks": ["grounded_args"]}}}
    e = engine(client, r, audit=AuditLog(log))
    d = e.evaluate("refund_order", {"order_id": "ORD-123456"}, convo(("user", "refund please")))
    assert d.action == "proceed" and d.would_have == "guide" and d.mode == "shadow"
    row = json.loads(log.read_text().splitlines()[0])
    assert row["would_have"] == "guide" and row["action"] == "proceed"


def test_shadow_mode_still_enforces_hard_rules(client):
    r = {"tools": {"t": {"mode": "shadow", "hard_rules": [{"arg": "n", "max": 1}], "checks": ["grounded_args"]}}}
    assert engine(client, r).evaluate("t", {"n": 5}, []).action == "deny"


def test_server_down_fails_closed(server):
    c = DeciderClient(server.url, timeout=1.0)
    server.set_mode("error")
    d = engine(c).evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD)
    assert d.action == "deny" and d.source == "decider_error"


def test_server_unreachable_can_fail_open_per_tool():
    c = DeciderClient("http://127.0.0.1:9", timeout=0.5)
    r = {"tools": {"t": {"on_decider_error": "proceed", "checks": ["grounded_args"]}}}
    d = engine(c, r).evaluate("t", {"a": 1}, [])
    assert d.action == "proceed" and d.source == "decider_error"


def test_circuit_breaker_skips_slow_calls_after_failures(server):
    now = [0.0]
    c = DeciderClient(server.url, timeout=1.0, breaker_threshold=2, breaker_cooldown=10, clock=lambda: now[0])
    server.set_mode("error")
    e = engine(c)
    for _ in range(2):
        e.evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD)
    assert c.breaker_open
    server.set_mode("ok")
    assert e.evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD).source == "decider_error"
    now[0] = 11.0  # cooldown over: recovers by itself
    assert e.evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD).action == "proceed"


def test_ladder_escalates_only_the_uncertain_band(client):
    seen = []

    def esc(ctx):
        seen.append(ctx.check.name)
        return True

    r = {"defaults": {"uncertain": "llm"}, "tools": {"t": {"checks": ["grounded_args"]}}}
    e = engine(client, r, escalator=esc)
    sure = e.evaluate("t", {"a": "hello"}, convo(("user", "hello")))
    assert sure.action == "proceed" and seen == []  # confident: no escalation
    unsure = e.evaluate("t", {"a": "hello", "b": "nope"}, convo(("user", "hello")))
    assert unsure.action == "proceed" and seen == ["grounded_args"]
    assert unsure.checks[0].escalated


@pytest.mark.parametrize("verdict, action", [(False, "guide"), (None, "guide")])
def test_ladder_reject_or_cannot_tell_guides(client, verdict, action):
    r = {"defaults": {"uncertain": "llm"}, "tools": {"t": {"checks": ["grounded_args"]}}}
    e = engine(client, r, escalator=lambda ctx: verdict)
    assert e.evaluate("t", {"a": "hello", "b": "nope"}, convo(("user", "hello"))).action == action


def test_escalator_crash_never_crashes_agent(client):
    def boom(ctx):
        raise RuntimeError("bedrock down")

    r = {"defaults": {"uncertain": "llm"}, "tools": {"t": {"checks": ["grounded_args"]}}}
    d = engine(client, r, escalator=boom).evaluate("t", {"a": "hello", "b": "nope"}, convo(("user", "hello")))
    assert d.action == "guide"


def test_llm_without_escalator_fails_at_startup(client):
    with pytest.raises(ValueError, match="escalator"):
        engine(client, {"defaults": {"uncertain": "llm"}, "tools": {"t": {"checks": ["grounded_args"]}}})


def test_choice_check(client):
    r = {"tools": {"t": {"checks": [{
        "name": "intent", "type": "choice", "question": "What is the request?",
        "criteria": {"refund": "wants money back", "complaint": "just unhappy"},
        "allow": ["refund"], "on_fail": "deny"}]}}}
    e = engine(client, r)
    assert e.evaluate("t", {}, convo(("user", "I want a refund"))).action == "proceed"
    assert e.evaluate("t", {}, convo(("user", "this is a complaint"))).action == "deny"
    assert e.evaluate("t", {}, convo(("user", "hmm"))).action == "guide"  # low confidence -> ladder


def test_audit_log_redaction_and_privacy(client, tmp_path):
    log = tmp_path / "a.jsonl"
    e = engine(client, audit=AuditLog(log, redact=["amount"]))
    e.evaluate("refund_order", {"order_id": "ORD-482913", "amount": 42.5}, GOOD)
    row = json.loads(log.read_text())
    assert row["arguments"]["amount"] == "***" and row["arguments"]["order_id"] == "ORD-482913"
    assert "context" not in row  # message text is off by default
    assert row["checks"][0]["name"] == "grounded_args"


def test_transcript_budget_keeps_newest_and_roles():
    from deciderkit.conversation import messages_to_transcript

    msgs = [{"role": "user", "content": [{"text": f"message {i} " + "x" * 500}]} for i in range(10)]
    msgs.append({"role": "user", "content": [{"toolResult": {"content": [{"json": {"id": "ORD-1"}}]}}]})
    t = messages_to_transcript(msgs, max_total_chars=1200)
    assert sum(len(m["text"]) for m in t) <= 1200 + 600  # budget, plus at most the forced newest message
    assert t[-1]["role"] == "tool_result" and "ORD-1" in t[-1]["text"]  # newest always kept
    assert "message 0" not in " ".join(m["text"] for m in t)  # oldest dropped first


def test_ladder_asks_the_bigger_model_once_per_call_not_once_per_check(client):
    seen = []

    def esc(ctx):
        seen.append(ctx.uncertain_checks)
        return True

    r = {"defaults": {"uncertain": "llm"},
         "tools": {"t": {"checks": ["grounded_args", "intent_matches_tool"]}}}
    d = engine(client, r, escalator=esc).evaluate("t", {"a": "hello", "b": "nope"}, convo(("user", "hello")))
    assert len(seen) == 1 and set(seen[0]) == {"grounded_args", "intent_matches_tool"}
    assert d.action == "proceed" and all(c.escalated for c in d.checks)
