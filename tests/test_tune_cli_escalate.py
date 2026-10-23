import json

from deciderkit.cli import main
from deciderkit.escalate import BedrockEscalator
from deciderkit.tune import tune


def synthetic():
    # confident-good, confident-bad, and a noisy middle that must NOT be auto-decided
    good = [(0.97, True)] * 30 + [(0.90, True)] * 20
    bad = [(0.03, False)] * 30 + [(0.10, False)] * 20
    mid = [(0.5, True), (0.5, False), (0.6, False), (0.4, True), (0.55, True), (0.45, False)]
    return good + bad + mid


def test_tune_finds_safe_thresholds_and_leaves_middle_alone():
    r = tune(synthetic())
    assert r.allow_above is not None and 0.6 < r.allow_above <= 0.90
    assert r.block_below is not None and r.block_below <= 0.5
    assert r.wrong_auto_proceeds == 0  # wrongly letting a bad call through is the costly error
    assert r.auto_proceed_precision >= 0.99 and r.auto_block_precision >= 0.95
    assert 0.9 < r.auto_decided < 1.0  # most decided automatically, the noisy middle is not


def test_tune_refuses_when_data_is_noise():
    noisy = [(0.5, True), (0.5, False)] * 20
    r = tune(noisy)
    assert r.allow_above is None and r.block_below is None


def test_tune_empty():
    assert tune([]).n == 0


def test_cli_validate_ok_and_bad(tmp_path, capsys):
    assert main(["validate", "recipes/support_ops.yaml"]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("tools: {}\n")
    assert main(["validate", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_cli_tune_and_summarize(tmp_path, capsys):
    f = tmp_path / "labeled.jsonl"
    rows = [{"p": p, "label": ok} for p, ok in synthetic()]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["tune", str(f)]) == 0
    out = capsys.readouterr().out
    assert "allow_above" in out and "block_below" in out
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"tool": "t", "action": "proceed", "would_have": "guide", "latency_ms": 12.0}) + "\n")
    assert main(["summarize", str(log)]) == 0
    assert "would have blocked" in capsys.readouterr().out


def test_cli_tune_reads_audit_rows_with_labels(tmp_path):
    f = tmp_path / "a.jsonl"
    rows = [{"label": ok, "checks": [{"name": "grounded_args", "value": p}]} for p, ok in synthetic()]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["tune", str(f), "--check", "grounded_args"]) == 0


class FakeBedrock:
    def __init__(self, text):
        self.text = text

    def converse(self, **kw):
        assert kw["inferenceConfig"]["temperature"] == 0
        return {"output": {"message": {"content": [{"text": self.text}]}},
                "usage": {"inputTokens": 120, "outputTokens": 8}}


def test_bedrock_escalator_parses_and_counts_tokens():
    esc = BedrockEscalator("some-model", client=FakeBedrock('Sure: {"supported": true}'))
    assert esc.judge([], "t", {}) is True
    assert (esc.calls, esc.input_tokens, esc.output_tokens) == (1, 120, 8)
    assert BedrockEscalator("m", client=FakeBedrock('{"supported": false}')).judge([], "t", {}) is False
    assert BedrockEscalator("m", client=FakeBedrock("no json here")).judge([], "t", {}) is None
    assert BedrockEscalator("m", client=FakeBedrock('{"supported": "yes"}')).judge([], "t", {}) is None


def test_rows_use_per_check_truth_when_available():
    from deciderkit.tune import rows_from_log

    rows = [{"label": False, "labels": {"grounded_args": True, "intent_matches_tool": False},
             "checks": [{"name": "grounded_args", "value": 0.9}, {"name": "intent_matches_tool", "value": 0.1}]}]
    assert rows_from_log(rows, "grounded_args") == [(0.9, True)]  # bad call, but not bad for this check
    assert rows_from_log(rows, "intent_matches_tool") == [(0.1, False)]
