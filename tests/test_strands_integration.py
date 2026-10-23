"""End-to-end through the REAL Strands Agent and its intervention system, with a scripted fake model.

The fake model proposes one tool call, then ends the turn after seeing the tool result.
No network, no Bedrock, no Decider model: the Decider server is the mock.
"""

import json

import pytest

pytest.importorskip("strands.interventions")

from strands import Agent, tool
from strands.models import Model

from deciderkit import DeciderGuard

EXECUTED: list[dict] = []


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund an order."""
    EXECUTED.append({"order_id": order_id, "amount": amount})
    return f"refunded {amount} on {order_id}"


class ScriptedModel(Model):
    def __init__(self, call_input):
        self.call_input = call_input
        self.tool_results_seen: list[str] = []

    def update_config(self, **kw): ...
    def get_config(self):
        return {}

    async def structured_output(self, *a, **kw):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        last = messages[-1]
        results = [b for b in last["content"] if "toolResult" in b]
        yield {"messageStart": {"role": "assistant"}}
        if results:
            self.tool_results_seen.append(json.dumps(results[0]["toolResult"]))
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "refund_order"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.call_input)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                            "metrics": {"latencyMs": 1}}}


def run(call_input, user_text, client, **guard_kw):
    EXECUTED.clear()
    model = ScriptedModel(call_input)
    guard = DeciderGuard("recipes/support_ops.yaml", client=client, **guard_kw)
    agent = Agent(model=model, tools=[refund_order], interventions=[guard], callback_handler=None)
    agent(user_text)
    return model, guard


def test_guessed_arguments_never_reach_the_tool(client):
    model, guard = run({"order_id": "ORD-123456", "amount": 20.0}, "refund my order", client)
    assert EXECUTED == []
    assert guard.last_decision.action == "guide"
    assert "Do not guess" in model.tool_results_seen[0]  # the model was told why


def test_grounded_call_runs(client):
    text = "I want a refund of 42.50 on ORD-482913, it arrived broken"
    model, guard = run({"order_id": "ORD-482913", "amount": 42.5}, text, client)
    assert EXECUTED == [{"order_id": "ORD-482913", "amount": 42.5}]
    assert guard.last_decision.action == "proceed"


def test_hard_rule_blocks_even_when_user_asked(client):
    text = "I want a refund of 900 on ORD-482913"
    model, guard = run({"order_id": "ORD-482913", "amount": 900.0}, text, client)
    assert EXECUTED == [] and guard.last_decision.source == "hard_rule"
    assert "human agent" in model.tool_results_seen[0]


def test_decider_outage_blocks_risky_tool(server):
    from deciderkit import DeciderClient

    server.set_mode("error")
    c = DeciderClient(server.url, timeout=1.0)
    text = "I want a refund of 42.50 on ORD-482913"
    model, guard = run({"order_id": "ORD-482913", "amount": 42.5}, text, c)
    assert EXECUTED == [] and guard.last_decision.source == "decider_error"


def test_audit_file_written(client, tmp_path):
    log = tmp_path / "audit.jsonl"
    run({"order_id": "ORD-123456", "amount": 20.0}, "refund my order", client, audit=str(log))
    row = json.loads(log.read_text().splitlines()[0])
    assert row["tool"] == "refund_order" and row["action"] == "guide"


def test_confirm_action_maps_to_strands_confirm(client):
    from types import SimpleNamespace

    from strands.interventions import Confirm, Deny, Guide, Proceed

    r = {"defaults": {"uncertain": "confirm"}, "tools": {"t": {"checks": ["grounded_args"]}}}
    guard = DeciderGuard(r, client=client)
    agent = SimpleNamespace(messages=[{"role": "user", "content": [{"text": "hello"}]}])

    def act(args):
        ev = SimpleNamespace(agent=agent, tool_use={"name": "t", "input": args})
        return guard.before_tool_call(ev)

    assert isinstance(act({"a": "hello"}), Proceed)
    assert isinstance(act({"a": "hello", "b": "nope"}), Confirm)  # unsure -> human approval
    assert isinstance(act({"b": "nope"}), Guide)  # clearly guessed -> guide
    strict = DeciderGuard({"tools": {"t": {"checks": [{"use": "grounded_args", "on_fail": "deny"}]}}}, client=client)
    ev = SimpleNamespace(agent=agent, tool_use={"name": "t", "input": {"b": "nope"}})
    assert isinstance(strict.before_tool_call(ev), Deny)
