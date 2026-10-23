import pytest

from deciderkit import RecipeError, load_recipe, with_mode

BASE = {"tools": {"t": {"checks": ["grounded_args"]}}}


def test_loads_demo_recipe():
    r = load_recipe("recipes/support_ops.yaml")
    assert set(r.tools) == {"lookup_orders", "refund_order", "cancel_order", "update_address"}
    assert r.tools["lookup_orders"].skip
    refund = r.tools["refund_order"]
    assert [c.name for c in refund.checks] == ["grounded_args", "intent_matches_tool"]
    assert len(refund.hard_rules) == 5


def test_builtin_shorthand_and_overrides():
    r = load_recipe({"tools": {"t": {"checks": [{"use": "grounded_args", "allow_above": 0.9, "on_fail": "deny"}]}}})
    c = r.tools["t"].checks[0]
    assert (c.allow_above, c.on_fail, c.type) == (0.9, "deny", "noul")


def test_defaults_flow_into_checks():
    r = load_recipe({"defaults": {"allow_above": 0.7, "block_below": 0.2, "uncertain": "deny"}, **BASE})
    assert r.tools["t"].uncertain == "deny"
    assert (r.tools["t"].checks[0].allow_above, r.tools["t"].checks[0].block_below) == (0.7, 0.2)


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"tools": {}}, "non-empty"),
        ({"toolz": {}}, "unknown key"),
        ({"version": 2, **BASE}, "version"),
        ({"tools": {"t": {"checks": ["nope"]}}}, "unknown built-in"),
        ({"tools": {"t": {"mode": "yolo"}}}, "not one of"),
        ({"defaults": {"allow_above": 0.2, "block_below": 0.5}, **BASE}, "lower than"),
        ({"defaults": {"allow_above": 1.5}, **BASE}, "between 0 and 1"),
        ({"tools": {"t": {"hard_rules": [{"arg": "a"}]}}}, "exactly one"),
        ({"tools": {"t": {"hard_rules": [{"arg": "a", "pattern": "("}]}}}, "invalid regex"),
        ({"tools": {"t": {"checks": [{"name": "x", "type": "noul"}]}}}, "question"),
        ({"tools": {"t": {"checks": [{"name": "x", "type": "choice", "question": "q",
                                      "criteria": {"a": "1", "b": "2"}, "allow": ["z"]}]}}}, "allow"),
        ({"tools": {"t": {"checks": ["grounded_args", "grounded_args"]}}}, "duplicate"),
        ({"tools": {"t": {"checks": [{"use": "grounded_args", "on_fail": "explode"}]}}}, "not one of"),
    ],
)
def test_invalid_recipes_fail_loudly(bad, fragment):
    with pytest.raises(RecipeError) as e:
        load_recipe(bad)
    assert fragment in str(e.value)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(RecipeError):
        load_recipe(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("tools: [unclosed")
    with pytest.raises(RecipeError):
        load_recipe(bad)


def test_with_mode_forces_every_tool():
    r = with_mode(load_recipe("recipes/support_ops.yaml"), "shadow")
    assert all(t.mode == "shadow" for t in r.tools.values())
