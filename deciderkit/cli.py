"""Command line: ``deciderkit validate | tune | breakdown | summarize | health | mock-server``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .audit import read_log
from .recipe import RecipeError, load_recipe
from .tune import rows_from_log, tune


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(recipe.tools)} tool(s), unlisted_tools={recipe.unlisted_tools}")
    for t in recipe.tools.values():
        if t.skip:
            print(f"  {t.name}: skipped")
        else:
            checks = ", ".join(c.name for c in t.checks) or "none"
            print(f"  {t.name}: mode={t.mode}, {len(t.hard_rules)} hard rule(s), checks=[{checks}], uncertain={t.uncertain}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    rows = read_log(args.file)
    data = rows_from_log(rows, args.check)
    if not data:
        print("No labeled rows found. Each row needs a boolean 'label' and a 'p' (or a check value).", file=sys.stderr)
        return 1
    res = tune(data, proceed_precision=args.proceed_precision, block_precision=args.block_precision)
    print(f"Labeled rows: {res.n}")
    if res.note:
        print(f"Note: {res.note}")
    if res.allow_above is None and res.block_below is None:
        print("No safe thresholds found at these targets. Add more labeled rows or lower the targets.")
        return 2
    print("Suggested recipe lines:")
    if res.allow_above is not None:
        print(f"  allow_above: {res.allow_above}   # auto-proceed precision {res.auto_proceed_precision:.3f}, "
              f"{res.wrong_auto_proceeds} wrong")
    if res.block_below is not None:
        print(f"  block_below: {res.block_below}   # auto-block precision {res.auto_block_precision:.3f}, "
              f"{res.wrong_auto_blocks} wrong")
    print(f"Decided without escalation: {res.auto_decided:.1%} of rows")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    rows = read_log(args.file)
    if not rows:
        print("Log is empty.")
        return 0
    actions = Counter(r.get("action") for r in rows)
    would = Counter(r.get("would_have") for r in rows if r.get("would_have"))
    by_tool = Counter(r.get("tool") for r in rows)
    lat = sorted(r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float)))
    print(f"Decisions: {len(rows)}")
    print("Actions taken:", dict(actions))
    if would:
        print("Shadow mode would have blocked:", dict(would))
    print("By tool:", dict(by_tool))
    if lat:
        print(f"Latency ms: p50={lat[len(lat) // 2]:.1f}  p95={lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.1f}")
    return 0


def _cmd_breakdown(args: argparse.Namespace) -> int:
    """Per-category view of benchmark rows: which kinds of cases is Decider unsure about?"""
    rows = read_log(args.file)
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r.get("category", "?"), []).append(r)
    names = sorted({c["name"] for r in rows for c in r.get("checks", [])})
    for check in names:
        print(f"\n== {check}  (thresholds: pass >= {args.allow_above}, fail <= {args.block_below}) ==")
        print(f"{'category':26} {'n':>3} {'should':>6} {'pass':>5} {'unsure':>6} {'fail':>5} {'p min':>6} {'p mean':>6} {'p max':>6}")
        for cat, rs in sorted(cats.items()):
            vals = [(c["value"], r.get("labels", {}).get(check, r["label"]))
                    for r in rs for c in r.get("checks", []) if c["name"] == check and isinstance(c.get("value"), (int, float))]
            if not vals:
                continue
            ps = [v for v, _ in vals]
            ok = sum(1 for v, _ in vals if v >= args.allow_above)
            bad = sum(1 for v, _ in vals if v <= args.block_below)
            should = "pass" if all(t for _, t in vals) else ("fail" if not any(t for _, t in vals) else "mixed")
            print(f"{cat:26} {len(vals):>3} {should:>6} {ok:>5} {len(vals) - ok - bad:>6} {bad:>5} "
                  f"{min(ps):>6.2f} {sum(ps) / len(ps):>6.2f} {max(ps):>6.2f}")
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    from .client import DeciderClient, DeciderUnavailable

    try:
        print(json.dumps(DeciderClient(args.url).health(), indent=2))
        return 0
    except DeciderUnavailable as exc:
        print(f"UNREACHABLE: {exc}", file=sys.stderr)
        return 1


def _cmd_mock(args: argparse.Namespace) -> int:
    from .testing.mock_server import serve

    serve(args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="deciderkit", description="Decision guardrails for Strands agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a recipe file")
    v.add_argument("recipe", type=Path)
    v.set_defaults(fn=_cmd_validate)

    t = sub.add_parser("tune", help="suggest thresholds from labeled rows")
    t.add_argument("file", type=Path)
    t.add_argument("--check", default=None, help="check name to read P(true) from (audit-log rows)")
    t.add_argument("--proceed-precision", type=float, default=0.99)
    t.add_argument("--block-precision", type=float, default=0.95)
    t.set_defaults(fn=_cmd_tune)

    s = sub.add_parser("summarize", help="summarize an audit log")
    s.add_argument("file", type=Path)
    s.set_defaults(fn=_cmd_summarize)

    b = sub.add_parser("breakdown", help="per-category view of benchmark rows")
    b.add_argument("file", type=Path)
    b.add_argument("--allow-above", type=float, default=0.80)
    b.add_argument("--block-below", type=float, default=0.30)
    b.set_defaults(fn=_cmd_breakdown)

    h = sub.add_parser("health", help="ping a Decider server")
    h.add_argument("--url", default=None)
    h.set_defaults(fn=_cmd_health)

    m = sub.add_parser("mock-server", help="run the fake Decider (testing only)")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8099)
    m.set_defaults(fn=_cmd_mock)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
