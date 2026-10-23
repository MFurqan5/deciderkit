"""The Strands adapter: one class you pass to ``Agent(interventions=[...])``."""

from __future__ import annotations

from typing import Any

from strands.interventions import Confirm, Deny, Guide, InterventionHandler, OnError, Proceed

from .audit import AuditLog
from .client import DeciderClient
from .conversation import messages_to_transcript
from .engine import Decision, DecisionEngine, Escalator
from .recipe import Recipe


class DeciderGuard(InterventionHandler):
    """Checks every tool call against a YAML recipe before it runs.

    Example::

        agent = Agent(tools=[refund_order], interventions=[DeciderGuard("recipe.yaml")])
    """

    name = "deciderkit"

    def __init__(
        self,
        recipe: str | dict | Recipe,
        *,
        client: DeciderClient | None = None,
        escalator: Escalator | None = None,
        audit: AuditLog | str | None = None,
        max_context_messages: int = 12,
        max_context_chars: int = 1800,
    ) -> None:
        if isinstance(audit, str):
            audit = AuditLog(audit)
        self.engine = DecisionEngine(recipe, client=client, escalator=escalator, audit=audit)
        self.max_context_messages = max_context_messages
        self.max_context_chars = max_context_chars
        self.last_decision: Decision | None = None

    @property
    def on_error(self) -> OnError:
        # If anything unexpected breaks inside the guard, block the call (fail closed).
        return "deny"

    def before_tool_call(self, event: Any, **kwargs: Any):  # noqa: ANN201
        tool_use = event.tool_use
        conversation = messages_to_transcript(
            getattr(event.agent, "messages", None),
            max_messages=self.max_context_messages,
            max_total_chars=self.max_context_chars,
        )
        decision = self.engine.evaluate(tool_use["name"], dict(tool_use.get("input") or {}), conversation)
        self.last_decision = decision
        if decision.action == "proceed":
            return Proceed(reason=decision.reason)
        if decision.action == "guide":
            return Guide(feedback=decision.feedback or decision.reason, reason=decision.reason)
        if decision.action == "confirm":
            return Confirm(
                prompt=f"Approve '{decision.tool}'? The safety check was unsure. ({decision.reason})",
                reason=decision.reason,
            )
        return Deny(reason=decision.feedback or decision.reason)
