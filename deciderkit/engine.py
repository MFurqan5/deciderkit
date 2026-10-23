"""The decision engine. Pure Python: no Strands import, so it is easy to test and reuse anywhere.

Order of work for each tool call:
  1. Tool not in the recipe?          -> ``unlisted_tools`` (proceed or deny)
  2. Hard rules (plain code)           -> any violation is an immediate deny
  3. Decider checks (one batched call) -> each check is pass / fail / uncertain
  4. Combine: fail beats uncertain beats pass
  5. Uncertain -> the "ladder": guide, deny, confirm, proceed, or ask a bigger model (``llm``)
  6. Shadow mode -> log what we WOULD have done, but let the call through
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .audit import AuditLog
from .client import DeciderClient, DeciderError
from .conversation import build_state
from .recipe import Check, HardRule, Recipe, ToolPolicy, load_recipe

DEFAULT_GUIDE = (
    "This action was paused because some details may not have come from the user. "
    "Ask the user to confirm or provide them, then try again."
)


@dataclass
class CheckOutcome:
    name: str
    verdict: str  # pass | fail | uncertain
    value: float | str | None = None  # P(true) for noul, chosen option for choice
    confidence: float | None = None
    escalated: bool = False
    latency_ms: float | None = None


@dataclass
class Decision:
    action: str  # proceed | guide | deny | confirm
    reason: str = ""
    feedback: str = ""
    tool: str = ""
    mode: str = "enforce"
    would_have: str | None = None  # in shadow mode: the action enforce mode would have taken
    source: str = ""  # unlisted | skip | hard_rule | decider | decider_error | escalation
    checks: list[CheckOutcome] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EscalationContext:
    tool: str
    arguments: dict[str, Any]
    conversation: list[dict[str, str]]
    check: Check  # the first uncertain check
    outcome: CheckOutcome
    uncertain_checks: tuple[str, ...] = ()  # names of every check that was uncertain


class Escalator(Protocol):
    """Anything callable that returns True (approve), False (reject) or None (cannot tell)."""

    def __call__(self, ctx: EscalationContext) -> bool | None: ...


def _check_hard_rule(rule: HardRule, args: dict[str, Any]) -> str | None:
    """Return an error message if the rule is violated, else None."""
    if rule.kind == "required":
        v = args.get(rule.arg)
        if v is None or (isinstance(v, str) and not v.strip()):
            return rule.message or f"argument '{rule.arg}' is required"
        return None
    if rule.arg not in args:
        return None  # only 'required' cares about missing arguments
    v = args[rule.arg]
    if rule.kind in ("min", "max"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return rule.message or f"argument '{rule.arg}' must be a number"
        if rule.kind == "min" and v < rule.value:
            return rule.message or f"'{rule.arg}' must be at least {rule.value}"
        if rule.kind == "max" and v > rule.value:
            return rule.message or f"'{rule.arg}' must be at most {rule.value}"
        return None
    if rule.kind == "pattern":
        if not isinstance(v, str) or not re.fullmatch(rule.value, v):
            return rule.message or f"'{rule.arg}' has an invalid format"
        return None
    if rule.kind == "one_of":
        if v not in rule.value:
            return rule.message or f"'{rule.arg}' must be one of {rule.value}"
        return None
    return None


def _question_payload(check: Check) -> dict[str, Any]:
    q: dict[str, Any] = {"type": check.type, "instructions": check.question}
    if check.criteria:
        q["criteria"] = dict(check.criteria)
    return q


def _interpret(check: Check, answer: dict[str, Any]) -> CheckOutcome:
    if check.type == "noul":
        p = float(answer["noul"])
        if p >= check.allow_above:
            verdict = "pass"
        elif p <= check.block_below:
            verdict = "fail"
        else:
            verdict = "uncertain"
        return CheckOutcome(check.name, verdict, round(p, 4), round(abs(2 * p - 1), 4))
    chosen = str(answer["choice"])
    conf = float(answer["confidence"])
    if conf < check.min_confidence:
        verdict = "uncertain"
    else:
        verdict = "pass" if chosen in check.allow else "fail"
    return CheckOutcome(check.name, verdict, chosen, round(conf, 4))


class DecisionEngine:
    def __init__(
        self,
        recipe: str | Any,
        *,
        client: DeciderClient | None = None,
        escalator: Escalator | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.recipe: Recipe = load_recipe(recipe)
        self.client = client or DeciderClient()
        self.escalator = escalator
        self.audit = audit
        if self.recipe.uses_action("llm") and escalator is None:
            raise ValueError(
                "recipe uses 'uncertain: llm' but no escalator was given. "
                "Pass escalator=BedrockEscalator(...) or your own callable."
            )

    # ------------------------------------------------------------------
    def evaluate(
        self, tool: str, arguments: dict[str, Any], conversation: list[dict[str, str]]
    ) -> Decision:
        start = time.perf_counter()
        decision = self._evaluate(tool, arguments or {}, conversation)
        decision.tool = tool
        decision.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.audit:
            self.audit.record(self._audit_entry(decision, arguments, conversation))
        return decision

    def _evaluate(self, tool: str, args: dict[str, Any], conv: list[dict[str, str]]) -> Decision:
        policy = self.recipe.tools.get(tool)
        if policy is None:
            if self.recipe.unlisted_tools == "deny":
                return Decision("deny", f"tool '{tool}' is not in the recipe", source="unlisted")
            return Decision("proceed", "tool not in recipe", source="unlisted")
        if policy.skip:
            return Decision("proceed", "skipped by recipe", source="skip", mode=policy.mode)

        for rule in policy.hard_rules:  # hard rules always enforce, even in shadow mode
            err = _check_hard_rule(rule, args)
            if err:
                return Decision(
                    "deny", f"hard rule: {err}", source="hard_rule", mode=policy.mode,
                    feedback=f"Blocked by policy: {err}.",
                )

        if not policy.checks:
            return Decision("proceed", "no checks configured", source="skip", mode=policy.mode)

        raw = self._run_checks(policy, tool, args, conv)
        raw.mode = policy.mode
        if policy.mode == "shadow" and raw.action != "proceed":
            raw.would_have = raw.action
            raw.action = "proceed"
            raw.reason = f"shadow mode (would have: {raw.would_have}) - {raw.reason}"
        return raw

    # ------------------------------------------------------------------
    def _run_checks(
        self, policy: ToolPolicy, tool: str, args: dict[str, Any], conv: list[dict[str, str]]
    ) -> Decision:
        questions = {c.name: _question_payload(c) for c in policy.checks}
        try:
            result = self.client.ask(build_state(tool, args, conv), questions)
        except DeciderError as exc:
            return self._on_error(policy, str(exc))
        try:
            outcomes = [
                self._with_latency(_interpret(c, result.answers[c.name]), result.server_latency_ms)
                for c in policy.checks
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return self._on_error(policy, f"unusable Decider answer: {exc}")

        by_name = {c.name: c for c in policy.checks}
        if policy.uncertain == "llm":
            unsure_now = [o for o in outcomes if o.verdict == "uncertain"]
            if unsure_now:  # ONE second opinion per tool call, applied to every uncertain check
                self._escalate(unsure_now, by_name, tool, args, conv)

        fails = [o for o in outcomes if o.verdict == "fail"]
        if fails:
            checks = [by_name[o.name] for o in fails]
            action = "deny" if any(c.on_fail == "deny" for c in checks) else "guide"
            msg = " ".join(dict.fromkeys(c.message for c in checks if c.message)) or DEFAULT_GUIDE
            label = "second opinion rejected" if any(o.escalated for o in fails) else "failed"
            return Decision(
                action, f"{label}: " + ", ".join(o.name for o in fails), feedback=msg,
                source="decider", checks=outcomes,
            )
        unsure = [o for o in outcomes if o.verdict == "uncertain"]
        if unsure:
            # With 'llm' the escalator already ran; anything still uncertain falls back to guide.
            action = "guide" if policy.uncertain == "llm" else policy.uncertain
            return Decision(
                action, "uncertain: " + ", ".join(o.name for o in unsure),
                feedback=DEFAULT_GUIDE, source="decider", checks=outcomes,
            )
        helped = [o.name for o in outcomes if o.escalated]
        reason = ("unsure, second opinion approved: " + ", ".join(helped)) if helped else "all checks passed"
        return Decision("proceed", reason, source="decider", checks=outcomes)

    @staticmethod
    def _with_latency(outcome: CheckOutcome, ms: float | None) -> CheckOutcome:
        outcome.latency_ms = ms
        return outcome

    def _escalate(
        self, unsure: list[CheckOutcome], by_name: dict[str, Check], tool: str, args: dict[str, Any],
        conv: list[dict[str, str]],
    ) -> None:
        first = unsure[0]
        ctx = EscalationContext(tool, args, conv, by_name[first.name], first, tuple(o.name for o in unsure))
        try:
            verdict = self.escalator(ctx)  # type: ignore[misc]
        except Exception:  # an escalator crash must never crash the agent; stay uncertain
            return
        if verdict is True:
            for o in unsure:
                o.verdict, o.escalated = "pass", True
        elif verdict is False:
            for o in unsure:
                o.verdict, o.escalated = "fail", True

    def _on_error(self, policy: ToolPolicy, detail: str) -> Decision:
        action = policy.on_decider_error
        return Decision(
            action,
            f"Decider unavailable ({detail})",
            feedback="The safety check is temporarily unavailable, so this action was not run. Try again shortly."
            if action != "proceed"
            else "",
            source="decider_error",
        )

    def _audit_entry(
        self, d: Decision, arguments: dict[str, Any], conv: list[dict[str, str]]
    ) -> dict[str, Any]:
        last_user = next((m["text"] for m in reversed(conv) if m["role"] == "user"), "")
        return {
            "tool": d.tool,
            "arguments": arguments,
            "mode": d.mode,
            "action": d.action,
            "would_have": d.would_have,
            "source": d.source,
            "reason": d.reason,
            "latency_ms": d.latency_ms,
            "checks": [
                {
                    "name": c.name, "verdict": c.verdict, "value": c.value,
                    "confidence": c.confidence, "escalated": c.escalated, "latency_ms": c.latency_ms,
                }
                for c in d.checks
            ],
            "context": last_user[:300],
        }
