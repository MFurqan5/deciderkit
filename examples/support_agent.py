"""Demo: a support-ops agent whose risky tools are guarded by DeciderKit.

Setup (see README):   pip install -e ".[strands,bedrock]"
Run the Decider server somewhere (AWS GPU box, or `deciderkit mock-server` for a dry run), then:

  python examples/support_agent.py --decider-url http://HOST:8099            # guarded (default)
  python examples/support_agent.py --no-guard                                # the "before": agent may guess
  python examples/support_agent.py --shadow                                  # log only, change nothing
  python examples/support_agent.py --demo                                    # scripted prompts for the video

Needs AWS credentials with Amazon Bedrock model access for the agent's LLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from strands import Agent, tool
from strands.models import BedrockModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deciderkit import AuditLog, DeciderClient, DeciderGuard, load_recipe, with_mode  # noqa: E402
from deciderkit.escalate import BedrockEscalator  # noqa: E402
from deciderkit.recipe import replace_uncertain  # noqa: E402

# --- a tiny fake order database (no real money moves) ---------------------------------------
ORDERS = {
    "ORD-482913": {"email": "sara@example.com", "item": "headphones", "total": 42.50, "status": "delivered"},
    "ORD-110774": {"email": "sara@example.com", "item": "desk lamp", "total": 31.00, "status": "shipped"},
    "ORD-905531": {"email": "omar@example.com", "item": "running shoes", "total": 89.99, "status": "delivered"},
}
REFUNDED: list[str] = []


@tool
def lookup_orders(email: str) -> str:
    """List a customer's orders by email address."""
    rows = [f"{oid} - {o['item']} - total {o['total']:.2f} - {o['status']}"
            for oid, o in ORDERS.items() if o["email"] == email.lower()]
    return "Orders: " + "; ".join(rows) if rows else "No orders found."


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund money on an order. This moves money, so the user must have asked for it."""
    if order_id not in ORDERS:
        return f"REFUND FAILED: {order_id} does not exist."
    REFUNDED.append(order_id)
    return f"Refunded {amount:.2f} on {order_id}."


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that has not been delivered."""
    if order_id not in ORDERS:
        return f"CANCEL FAILED: {order_id} does not exist."
    ORDERS[order_id]["status"] = "cancelled"
    return f"Cancelled {order_id}."


@tool
def update_address(order_id: str, new_address: str) -> str:
    """Change the delivery address of an order."""
    return f"Address for {order_id} set to {new_address}."


SYSTEM_PROMPT = (
    "You are a customer-support agent for an online shop. Use the tools to help customers. "
    "Be brief and friendly. If you are missing a detail, you may ask the customer."
)

DEMO_PROMPTS = [
    "Hi, my headphones broke. Please refund my order.",  # no order id: a guard should stop a guess
    "My email is sara@example.com. Can you show my orders?",
    "Refund the full amount of ORD-482913, the headphones arrived broken.",  # grounded: should run
]


def show(entry: dict) -> None:
    mark = {"proceed": "ALLOW", "guide": "ASK  ", "deny": "BLOCK", "confirm": "CONFIRM"}.get(entry["action"], "?")
    extra = f" (shadow: would have {entry['would_have']})" if entry.get("would_have") else ""
    checks = ", ".join(f"{c['name']}={c['value']}" for c in entry["checks"])
    print(f"   [guard] {mark} {entry['tool']}{extra}  {entry['reason']}  [{checks}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default=str(ROOT / "recipes" / "support_ops.yaml"))
    p.add_argument("--decider-url", default=None, help="default: $STRANDS_DECIDER_URL or http://localhost:8099")
    p.add_argument("--model-id", default=None, help="Bedrock model id (default: the Strands default)")
    p.add_argument("--region", default=None)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--ladder", action="store_true",
                   help="unsure cases go to a bigger Bedrock model (needs --model-id) instead of asking the user")
    p.add_argument("--shadow", action="store_true", help="log what the guard would do but never block")
    p.add_argument("--audit", default=str(ROOT / "audit.jsonl"))
    p.add_argument("--demo", action="store_true", help="run scripted prompts instead of a chat")
    args = p.parse_args()

    interventions = []
    if not args.no_guard:
        recipe = load_recipe(args.recipe)
        if args.shadow:
            recipe = with_mode(recipe, "shadow")
        client = DeciderClient(args.decider_url)
        try:
            print("Decider server:", client.health())
        except Exception as exc:  # fail early and clearly
            sys.exit(f"Cannot reach the Decider server: {exc}")
        escalator = None
        if args.ladder:
            if not args.model_id:
                sys.exit("--ladder needs --model-id (the Bedrock model that gives the second opinion)")
            recipe = replace_uncertain(recipe, "llm")
            escalator = BedrockEscalator(args.model_id, region=args.region)
        interventions.append(
            DeciderGuard(recipe, client=client, escalator=escalator, audit=AuditLog(args.audit, sink=show))
        )

    kwargs = {"region_name": args.region} if args.region else {}
    if args.model_id:
        kwargs["model_id"] = args.model_id
    agent = Agent(
        model=BedrockModel(**kwargs),
        tools=[lookup_orders, refund_order, cancel_order, update_address],
        system_prompt=SYSTEM_PROMPT,
        interventions=interventions,
    )
    mode = "NO GUARD" if args.no_guard else ("shadow" if args.shadow else "guarded")
    if args.ladder and not args.no_guard:
        mode += " + ladder"
    print(f"Support agent ready ({mode}). Type 'quit' to exit.\n")

    prompts = iter(DEMO_PROMPTS) if args.demo else None
    while True:
        try:
            text = next(prompts) if prompts else input("you> ").strip()
        except (StopIteration, EOFError):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        if prompts:
            print(f"you> {text}")
        agent(text)
        print()
    print(f"Refunds issued this session: {REFUNDED or 'none'}")


if __name__ == "__main__":
    main()
