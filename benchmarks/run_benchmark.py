"""Compare three ways to gate tool calls on benchmarks/cases.jsonl.

  decider : Decider alone (unsure -> block/guide, which counts as "stopped")
  llm     : a Bedrock LLM judges every call
  ladder  : Decider first; only the unsure middle band goes to the LLM

Real run (needs a Decider server + AWS credentials with Bedrock access):
  python benchmarks/run_benchmark.py --decider-url http://HOST:8099 --llm-model-id <bedrock model id> \
      --llm-input-price 0.80 --llm-output-price 4.00 --hardware "g5.xlarge A10G"

Harness check only (NOT real numbers; uses the fake Decider and a fake LLM):
  python benchmarks/run_benchmark.py --mock
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from deciderkit import DeciderClient, DecisionEngine, load_recipe  # noqa: E402
from deciderkit.engine import EscalationContext  # noqa: E402
from deciderkit.escalate import BedrockEscalator  # noqa: E402
from deciderkit.recipe import replace_uncertain  # noqa: E402


class FakeLLM:
    """Stand-in judge for --mock: supported only if every argument value appears in the user text."""

    calls = input_tokens = output_tokens = 0

    def judge(self, conversation, tool, arguments):
        from deciderkit.testing.mock_server import grounded_fraction

        self.calls += 1
        self.input_tokens += 300
        self.output_tokens += 8
        state = {"conversation": conversation, "proposed_tool_call": {"arguments": arguments}}
        return grounded_fraction(state) == 1.0


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def score(name, cases, outcomes, latencies, llm_calls, tokens, prices):
    tp = sum(1 for c, o in zip(cases, outcomes) if c["label"] and o == "proceed")
    tn = sum(1 for c, o in zip(cases, outcomes) if not c["label"] and o != "proceed")
    fp = sum(1 for c, o in zip(cases, outcomes) if not c["label"] and o == "proceed")  # bad call let through
    fn = sum(1 for c, o in zip(cases, outcomes) if c["label"] and o != "proceed")  # good call blocked
    n = len(cases)
    cost = None
    if prices[0] is not None and prices[1] is not None:
        cost = tokens[0] / 1e6 * prices[0] + tokens[1] / 1e6 * prices[1]
    return {
        "setup": name, "n": n, "accuracy": (tp + tn) / n,
        "bad_calls_let_through": fp, "good_calls_blocked": fn,
        "p50_ms": statistics.median(latencies), "p95_ms": pct(latencies, 0.95),
        "llm_calls": llm_calls, "llm_cost_usd": cost,
    }


def run(args) -> dict:
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    recipe = load_recipe(args.recipe)
    server = None
    if args.mock:
        from deciderkit.testing.mock_server import MockDeciderServer

        server = MockDeciderServer().start()
        client = DeciderClient(server.url)
        llm = FakeLLM()
    else:
        client = DeciderClient(args.decider_url, timeout=args.timeout)
        if not args.llm_model_id:
            sys.exit("--llm-model-id is required for a real run (or use --mock)")
        llm = BedrockEscalator(args.llm_model_id, region=args.region)
    prices = (args.llm_input_price, args.llm_output_price)
    results = []
    rows: list[dict] = []
    try:
        health = client.health()

        # A) Decider alone
        eng = DecisionEngine(recipe, client=client)
        outs, lats = [], []
        uncertain = 0
        for c in cases:
            d = eng.evaluate(c["tool"], c["arguments"], c["conversation"])
            outs.append(d.action)
            lats.append(d.latency_ms)
            uncertain += d.reason.startswith("uncertain")
            rows.append({
                "id": c["id"], "category": c["category"], "label": c["label"],
                "checks": [{"name": o.name, "value": o.value} for o in d.checks],
                # truth per check: a bad case only counts as "bad" for the check that should catch it
                "labels": {o.name: bool(c["label"]) or c.get("bad_check") != o.name for o in d.checks},
            })
        r = score("decider", cases, outs, lats, 0, (0, 0), prices)
        r["sent_to_ladder_if_enabled"] = uncertain
        results.append(r)

        # B) LLM alone
        outs, lats = [], []
        c0, i0, o0 = llm.calls, llm.input_tokens, llm.output_tokens
        for c in cases:
            t = time.perf_counter()
            verdict = llm.judge(c["conversation"], c["tool"], c["arguments"])
            lats.append((time.perf_counter() - t) * 1000)
            outs.append("proceed" if verdict is True else "guide")
        results.append(score("llm", cases, outs, lats, llm.calls - c0,
                             (llm.input_tokens - i0, llm.output_tokens - o0), prices))

        # C) Ladder: Decider, then the LLM only for the unsure middle band
        ladder = DecisionEngine(replace_uncertain(recipe, "llm"), client=client, escalator=llm)
        outs, lats = [], []
        c0, i0, o0 = llm.calls, llm.input_tokens, llm.output_tokens
        for c in cases:
            d = ladder.evaluate(c["tool"], c["arguments"], c["conversation"])
            outs.append(d.action)
            lats.append(d.latency_ms)
        results.append(score("ladder", cases, outs, lats, llm.calls - c0,
                             (llm.input_tokens - i0, llm.output_tokens - o0), prices))
    finally:
        if server:
            server.stop()
    return {
        "mock": bool(args.mock), "hardware": args.hardware, "decider_health": health,
        "llm_model_id": None if args.mock else args.llm_model_id, "cases": len(cases), "results": results,
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    lines = []
    if report["mock"]:
        lines += ["> **MOCK RUN: these numbers are NOT Decider results. They only prove the harness works.**", ""]
    lines += [f"Hardware: {report['hardware'] or 'not stated'}  |  LLM: {report['llm_model_id'] or 'n/a'}  |  "
              f"Cases: {report['cases']}", "",
              "| Setup | Accuracy | Bad calls let through | Good calls blocked | p50 ms | p95 ms | LLM calls | LLM cost |",
              "|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        cost = "n/a" if r["llm_cost_usd"] is None else f"${r['llm_cost_usd']:.4f}"
        lines.append(f"| {r['setup']} | {r['accuracy']:.1%} | {r['bad_calls_let_through']} | "
                     f"{r['good_calls_blocked']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['llm_calls']} | {cost} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--cases", default=str(ROOT / "cases.jsonl"), help="labeled cases file")
    p.add_argument("--recipe", default=str(ROOT / "bench_recipe.yaml"), help="recipe to benchmark")
    p.add_argument("--decider-url")
    p.add_argument("--llm-model-id")
    p.add_argument("--region")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--llm-input-price", type=float, help="USD per 1M input tokens (look it up, do not guess)")
    p.add_argument("--llm-output-price", type=float, help="USD per 1M output tokens")
    p.add_argument("--hardware", default="")
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()
    report = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "mock" if args.mock else time.strftime("run-%Y%m%d-%H%M%S")
    rows = report.pop("rows")
    (out / f"{stem}-rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (out / f"{stem}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
