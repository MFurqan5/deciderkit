"""Show Decider's verdict on a few hand-written tool calls (a reliable moment for your video).

  python examples/show_decisions.py --decider-url http://localhost:8099 --recipe recipes/support_ops_tuned.yaml
  python examples/show_decisions.py ... --ladder --model-id us.amazon.nova-2-lite-v1:0 --region us-east-1

Per check:  name=P(true):verdict   (verdict: pass / fail / uncertain; a trailing * means the bigger model decided)
Without --ladder, anything uncertain is shown as ASK USER. With --ladder, uncertain cases get a second opinion.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deciderkit import DeciderClient, DecisionEngine, load_recipe  # noqa: E402
from deciderkit.recipe import replace_uncertain  # noqa: E402

CASES = [
    ("User gave id and amount",
     [("user", "Please refund 42.50 on ORD-482913, it arrived broken.")],
     {"order_id": "ORD-482913", "amount": 42.5}),
    ("Agent invented the order id",
     [("user", "I want a refund, my headphones arrived broken.")],
     {"order_id": "ORD-123456", "amount": 42.5}),
    ("Agent swapped two digits",
     [("user", "Please refund 42.50 on ORD-482913.")],
     {"order_id": "ORD-482193", "amount": 42.5}),
    ("User only asked about policy",
     [("user", "What is your refund policy for ORD-482913? Do not do anything yet.")],
     {"order_id": "ORD-482913", "amount": 42.5}),
]

LABEL = {"proceed": "ALLOW", "guide": "ASK USER", "deny": "BLOCK", "confirm": "CONFIRM"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider-url", default=None)
    ap.add_argument("--recipe", default=str(ROOT / "recipes" / "support_ops_tuned.yaml"))
    ap.add_argument("--ladder", action="store_true", help="send unsure cases to a bigger Bedrock model")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    a = ap.parse_args()

    recipe, escalator = load_recipe(a.recipe), None
    if a.ladder:
        if not a.model_id:
            sys.exit("--ladder needs --model-id")
        from deciderkit.escalate import BedrockEscalator

        recipe, escalator = replace_uncertain(recipe, "llm"), BedrockEscalator(a.model_id, region=a.region)
    else:
        recipe = replace_uncertain(recipe, "guide")
    engine = DecisionEngine(recipe, client=DeciderClient(a.decider_url), escalator=escalator)

    def run(case):
        return engine.evaluate("refund_order", case[2], [{"role": r, "text": t} for r, t in case[1]])

    run(CASES[0])  # warm-up: the first call after idle is slower and would distort the table
    print(f"{'case':30} {'decision':9} {'ms':>6}   checks")
    for case in CASES:
        d = run(case)
        checks = "  ".join(
            f"{c.name}={float(c.value):.2f}:{c.verdict}{'*' if c.escalated else ''}" for c in d.checks
        )
        print(f"{case[0]:30} {LABEL.get(d.action, d.action):9} {d.latency_ms:6.0f}   {checks}")


if __name__ == "__main__":
    main()
