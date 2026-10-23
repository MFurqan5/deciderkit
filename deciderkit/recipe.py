"""Recipe = the YAML file a developer writes. Strictly validated so typos fail loudly at startup."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


class RecipeError(ValueError):
    """The recipe file is invalid."""


FAIL_ACTIONS = ("guide", "deny")
UNCERTAIN_ACTIONS = ("guide", "deny", "confirm", "llm", "proceed")
ERROR_ACTIONS = ("deny", "guide", "proceed")
MODES = ("shadow", "enforce")
UNLISTED = ("proceed", "deny")

# Reusable, ready-made questions. Use them by name: ``checks: [grounded_args]``.
BUILTIN_CHECKS: dict[str, dict[str, Any]] = {
    "grounded_args": {
        "type": "noul",
        "question": (
            "Look at the proposed tool call. Is EVERY argument value explicitly stated by the user "
            "or returned by an earlier tool result? Answer true only if nothing was guessed, "
            "invented, assumed or copied from the assistant's own earlier messages."
        ),
        "criteria": {
            "true": "Every argument value appears in a user message or a tool result.",
            "false": "At least one argument value was guessed, invented, or never provided.",
        },
        "message": (
            "I was about to use details the user never gave. Do not guess. "
            "Ask the user for the missing or unclear details, then try again."
        ),
        "on_fail": "guide",
    },
    "intent_matches_tool": {
        "type": "noul",
        "question": (
            "Did the user clearly ask for exactly this action (this tool, on this target)? "
            "Answer false if the user only asked a question, was undecided, or asked for something different."
        ),
        "criteria": {
            "true": "The user clearly requested this exact action.",
            "false": "The user did not clearly request this action.",
        },
        "message": (
            "The user has not clearly asked for this action yet. "
            "Confirm what they want in plain words before doing it."
        ),
        "on_fail": "guide",
    },
}


@dataclass(frozen=True)
class HardRule:
    kind: str  # required | min | max | pattern | one_of
    arg: str
    value: Any = None
    message: str | None = None


@dataclass(frozen=True)
class Check:
    name: str
    type: str  # noul | choice
    question: str
    criteria: dict[str, str] | None = None
    allow_above: float = 0.80
    block_below: float = 0.30
    allow: tuple[str, ...] = ()
    min_confidence: float = 0.60
    on_fail: str = "guide"
    message: str = ""


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    skip: bool = False
    mode: str = "enforce"
    uncertain: str = "guide"
    on_decider_error: str = "deny"
    hard_rules: tuple[HardRule, ...] = ()
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class Recipe:
    tools: dict[str, ToolPolicy]
    unlisted_tools: str = "proceed"
    version: int = 1

    def uses_action(self, action: str) -> bool:
        return any(t.uncertain == action for t in self.tools.values() if t.checks)


@dataclass
class _Defaults:
    mode: str = "enforce"
    uncertain: str = "guide"
    on_decider_error: str = "deny"
    allow_above: float = 0.80
    block_below: float = 0.30
    min_confidence: float = 0.60
    extra: dict[str, Any] = field(default_factory=dict)


_TOP_KEYS = {"version", "defaults", "unlisted_tools", "tools"}
_DEFAULT_KEYS = {"mode", "uncertain", "on_decider_error", "allow_above", "block_below", "min_confidence"}
_TOOL_KEYS = {"skip", "mode", "uncertain", "on_decider_error", "hard_rules", "checks"}
_CHECK_KEYS = {
    "use", "name", "type", "question", "criteria", "allow_above", "block_below",
    "allow", "min_confidence", "on_fail", "message",
}
_RULE_KEYS = {"arg", "min", "max", "pattern", "one_of", "required", "message"}


def _unknown(where: str, got: set[str], allowed: set[str]) -> None:
    extra = got - allowed
    if extra:
        raise RecipeError(f"{where}: unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}")


def _enum(where: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise RecipeError(f"{where}: {value!r} is not one of {list(allowed)}")
    return value


def _prob(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise RecipeError(f"{where}: must be a number between 0 and 1, got {value!r}")
    return float(value)


def _parse_rules(tool: str, raw: Any) -> tuple[HardRule, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RecipeError(f"tools.{tool}.hard_rules: must be a list")
    rules: list[HardRule] = []
    for i, item in enumerate(raw):
        where = f"tools.{tool}.hard_rules[{i}]"
        if not isinstance(item, dict):
            raise RecipeError(f"{where}: must be a mapping")
        _unknown(where, set(item), _RULE_KEYS)
        message = item.get("message")
        if "required" in item:
            req = item["required"]
            if not isinstance(req, list) or not all(isinstance(a, str) for a in req):
                raise RecipeError(f"{where}.required: must be a list of argument names")
            rules.extend(HardRule("required", a, None, message) for a in req)
            continue
        arg = item.get("arg")
        if not isinstance(arg, str) or not arg:
            raise RecipeError(f"{where}: needs 'arg' (argument name) or 'required'")
        kinds = [k for k in ("min", "max", "pattern", "one_of") if k in item]
        if len(kinds) != 1:
            raise RecipeError(f"{where}: give exactly one of min, max, pattern, one_of")
        kind = kinds[0]
        value = item[kind]
        if kind in ("min", "max") and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise RecipeError(f"{where}.{kind}: must be a number")
        if kind == "pattern":
            if not isinstance(value, str):
                raise RecipeError(f"{where}.pattern: must be a string")
            try:
                re.compile(value)
            except re.error as exc:
                raise RecipeError(f"{where}.pattern: invalid regex ({exc})") from exc
        if kind == "one_of" and (not isinstance(value, list) or not value):
            raise RecipeError(f"{where}.one_of: must be a non-empty list")
        rules.append(HardRule(kind, arg, value, message))
    return tuple(rules)


def _parse_check(tool: str, idx: int, raw: Any, d: _Defaults) -> Check:
    where = f"tools.{tool}.checks[{idx}]"
    if isinstance(raw, str):
        raw = {"use": raw}
    if not isinstance(raw, dict):
        raise RecipeError(f"{where}: must be a name or a mapping")
    _unknown(where, set(raw), _CHECK_KEYS)
    merged: dict[str, Any] = {}
    if "use" in raw:
        base = BUILTIN_CHECKS.get(raw["use"])
        if base is None:
            raise RecipeError(f"{where}.use: unknown built-in {raw['use']!r}; known: {sorted(BUILTIN_CHECKS)}")
        merged.update(base)
        merged["name"] = raw["use"]
    merged.update({k: v for k, v in raw.items() if k != "use"})
    name = merged.get("name")
    if not isinstance(name, str) or not name:
        raise RecipeError(f"{where}: needs a 'name'")
    ctype = _enum(f"{where}.type", merged.get("type"), ("noul", "choice"))
    question = merged.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RecipeError(f"{where}.question: required")
    criteria = merged.get("criteria")
    allow: tuple[str, ...] = ()
    if ctype == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
            raise RecipeError(f"{where}.criteria: for noul use keys 'true' and/or 'false'")
    else:
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise RecipeError(f"{where}.criteria: choice needs a mapping of at least 2 options")
        raw_allow = merged.get("allow")
        if not isinstance(raw_allow, list) or not raw_allow or not set(raw_allow) <= set(criteria):
            raise RecipeError(f"{where}.allow: must be a non-empty list of options from criteria")
        allow = tuple(raw_allow)
    allow_above = _prob(f"{where}.allow_above", merged.get("allow_above", d.allow_above))
    block_below = _prob(f"{where}.block_below", merged.get("block_below", d.block_below))
    if ctype == "noul" and not block_below < allow_above:
        raise RecipeError(f"{where}: block_below ({block_below}) must be lower than allow_above ({allow_above})")
    return Check(
        name=name,
        type=ctype,
        question=question.strip(),
        criteria=criteria,
        allow_above=allow_above,
        block_below=block_below,
        allow=allow,
        min_confidence=_prob(f"{where}.min_confidence", merged.get("min_confidence", d.min_confidence)),
        on_fail=_enum(f"{where}.on_fail", merged.get("on_fail", "guide"), FAIL_ACTIONS),
        message=str(merged.get("message", "")),
    )


def parse_recipe(data: Any) -> Recipe:
    if not isinstance(data, dict):
        raise RecipeError("recipe must be a mapping at the top level")
    _unknown("recipe", set(data), _TOP_KEYS)
    version = data.get("version", 1)
    if version != 1:
        raise RecipeError(f"unsupported recipe version {version!r} (this release reads version 1)")
    raw_defaults = data.get("defaults") or {}
    if not isinstance(raw_defaults, dict):
        raise RecipeError("defaults: must be a mapping")
    _unknown("defaults", set(raw_defaults), _DEFAULT_KEYS)
    d = _Defaults(
        mode=_enum("defaults.mode", raw_defaults.get("mode", "enforce"), MODES),
        uncertain=_enum("defaults.uncertain", raw_defaults.get("uncertain", "guide"), UNCERTAIN_ACTIONS),
        on_decider_error=_enum(
            "defaults.on_decider_error", raw_defaults.get("on_decider_error", "deny"), ERROR_ACTIONS
        ),
        allow_above=_prob("defaults.allow_above", raw_defaults.get("allow_above", 0.80)),
        block_below=_prob("defaults.block_below", raw_defaults.get("block_below", 0.30)),
        min_confidence=_prob("defaults.min_confidence", raw_defaults.get("min_confidence", 0.60)),
    )
    if not d.block_below < d.allow_above:
        raise RecipeError("defaults: block_below must be lower than allow_above")
    unlisted = _enum("unlisted_tools", data.get("unlisted_tools", "proceed"), UNLISTED)
    raw_tools = data.get("tools")
    if not isinstance(raw_tools, dict) or not raw_tools:
        raise RecipeError("tools: must be a non-empty mapping of tool name -> policy")
    tools: dict[str, ToolPolicy] = {}
    for tname, traw in raw_tools.items():
        if traw is None:
            traw = {}
        if not isinstance(traw, dict):
            raise RecipeError(f"tools.{tname}: must be a mapping")
        _unknown(f"tools.{tname}", set(traw), _TOOL_KEYS)
        checks_raw = traw.get("checks") or []
        if not isinstance(checks_raw, list):
            raise RecipeError(f"tools.{tname}.checks: must be a list")
        checks = tuple(_parse_check(tname, i, c, d) for i, c in enumerate(checks_raw))
        names = [c.name for c in checks]
        if len(names) != len(set(names)):
            raise RecipeError(f"tools.{tname}.checks: duplicate check names")
        tools[str(tname)] = ToolPolicy(
            name=str(tname),
            skip=bool(traw.get("skip", False)),
            mode=_enum(f"tools.{tname}.mode", traw.get("mode", d.mode), MODES),
            uncertain=_enum(f"tools.{tname}.uncertain", traw.get("uncertain", d.uncertain), UNCERTAIN_ACTIONS),
            on_decider_error=_enum(
                f"tools.{tname}.on_decider_error", traw.get("on_decider_error", d.on_decider_error), ERROR_ACTIONS
            ),
            hard_rules=_parse_rules(str(tname), traw.get("hard_rules")),
            checks=checks,
        )
    return Recipe(tools=tools, unlisted_tools=unlisted, version=version)


def load_recipe(source: str | Path | dict | Recipe) -> Recipe:
    """Load a recipe from a YAML path, a dict, or pass an existing Recipe through."""
    if isinstance(source, Recipe):
        return source
    if isinstance(source, dict):
        return parse_recipe(source)
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeError(f"cannot read recipe {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path}: invalid YAML: {exc}") from exc
    return parse_recipe(data)


def with_mode(recipe: Recipe, mode: str) -> Recipe:
    """Return a copy of the recipe with every tool forced into ``mode`` (handy for a --shadow flag)."""
    _enum("mode", mode, MODES)
    return replace(recipe, tools={n: replace(t, mode=mode) for n, t in recipe.tools.items()})


def replace_uncertain(recipe: Recipe, action: str) -> Recipe:
    """Return a copy where every tool's "unsure" action is ``action`` (used by the benchmark ladder)."""
    _enum("uncertain", action, UNCERTAIN_ACTIONS)
    return replace(recipe, tools={n: replace(t, uncertain=action) for n, t in recipe.tools.items()})
