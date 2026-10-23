"""Turn Strands agent messages into the small, honest transcript we show Decider."""

from __future__ import annotations

import json
from typing import Any


def _tool_result_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in block.get("content", []) or []:
        if isinstance(item, dict):
            if "text" in item:
                parts.append(str(item["text"]))
            elif "json" in item:
                parts.append(json.dumps(item["json"], sort_keys=True, default=str))
    return " ".join(parts)


def messages_to_transcript(
    messages: list[dict[str, Any]] | None,
    *,
    max_messages: int = 12,
    max_chars: int = 600,
    max_total_chars: int = 1800,
) -> list[dict[str, str]]:
    """Flatten Strands messages to ``[{"role", "text"}]``, newest messages kept first.

    Roles: ``user`` (what the person typed), ``assistant`` (the agent's own words, NOT evidence),
    ``tool_result`` (data a tool returned, which counts as evidence). Tool *calls* are skipped:
    the call being judged is passed to Decider separately.

    Size is capped on purpose: the Decider training tooling truncates each state to about 2000
    characters by default, so very long transcripts are likely outside what the model saw in training.
    ``max_total_chars`` is the budget for the whole conversation; the newest message is always kept.
    """
    flat: list[dict[str, str]] = []
    for message in messages or []:
        role = message.get("role", "")
        for block in message.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if "text" in block and role in ("user", "assistant"):
                text = str(block["text"]).strip()
                if text:
                    flat.append({"role": role, "text": text[:max_chars]})
            elif "toolResult" in block:
                text = _tool_result_text(block["toolResult"]).strip()
                if text:
                    flat.append({"role": "tool_result", "text": text[:max_chars]})
    kept: list[dict[str, str]] = []
    used = 0
    for item in reversed(flat[-max_messages:]):
        if kept and used + len(item["text"]) > max_total_chars:
            break
        kept.append(item)
        used += len(item["text"])
    return list(reversed(kept))


def build_state(
    tool_name: str, arguments: dict[str, Any], conversation: list[dict[str, str]]
) -> dict[str, Any]:
    """The structured ``state`` sent to Decider (the server renders dicts deterministically)."""
    return {
        "conversation": conversation,
        "proposed_tool_call": {"tool": tool_name, "arguments": arguments},
        "evidence_rule": "Only 'user' and 'tool_result' messages are evidence. 'assistant' messages are not.",
    }
