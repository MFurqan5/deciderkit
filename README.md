# DeciderKit

**Decision guardrails for Strands agents, written as a YAML file.**
Built for the AWS Agent Experience Hackathon: Strands Decider. Tags: `#strands-decider-2b` `#guardrails`

> Status: experimental. Strands Decider 2B is an experimental model, and so is this kit. The 60 unit and integration tests use a fake Decider server and a scripted model. Real-model numbers come from the benchmark you run yourself (see below).

## The problem

An agent that can refund money, cancel orders or send email will sometimes **guess** a missing detail (which order? how much?) instead of asking. The mistake happens before any human sees it.

## The fix, in three steps

1. Write a recipe (YAML): which tools to check, and what to ask Decider about each.
2. Add one line to your agent.
3. Before every tool call, Decider answers a typed question in one fast call. The agent proceeds, asks the user, or escalates.

```yaml
# recipe.yaml
tools:
  refund_order:
    hard_rules:
      - { arg: amount, max: 500 }          # plain code, always runs first
    checks:
      - grounded_args                      # built-in: did the user really give these values?
```

```python
from strands import Agent
from deciderkit import DeciderGuard

agent = Agent(tools=[refund_order], interventions=[DeciderGuard("recipe.yaml")])
```

## Quickstart (no GPU needed)

```bash
pip install -e ".[strands,bedrock]"
deciderkit validate recipes/support_ops.yaml
deciderkit mock-server &                 # FAKE Decider for development only
python examples/support_agent.py --demo  # needs AWS credentials with Bedrock access
```

For the real model, see [docs/aws-setup.md](docs/aws-setup.md).

## The reusable decision pattern (what you can copy)

**Pattern: grounded tool arguments** (tag: `#guardrails`)

| Part | What it is |
|---|---|
| Typed question | `noul` (yes/no, returns P(true)): "Is every argument value stated by the user or returned by a tool result?" The text is in `deciderkit/recipe.py` (`BUILTIN_CHECKS`). |
| Option set | pass if P(true) at or above `allow_above`; fail if at or below `block_below`; the band between is "uncertain". For `choice` questions: a list of allowed options plus a minimum confidence. |
| Wiring | `DeciderGuard`, a Strands `InterventionHandler` that runs in `before_tool_call`. |
| Result | `Proceed`, `Guide` (tell the model why and let it ask), `Deny`, or `Confirm` (human approval). |
| How to reuse | Copy `recipes/support_ops.yaml`, rename your tools, keep or edit the built-in checks. No Python needed. |

A second built-in, `intent_matches_tool`, asks "did the user clearly ask for exactly this action?".

## How each tool call is decided

1. Tool not in the recipe: `unlisted_tools` (proceed or deny).
2. **Hard rules** (min, max, regex, one_of, required): plain code, always enforced, even in shadow mode.
3. **Decider checks**: all questions for the tool go in one request to `/v1/systemone`.
4. Each check is pass, fail or uncertain. Fail beats uncertain beats pass.
5. **Uncertain** follows `uncertain:` in the recipe: `guide`, `deny`, `confirm`, `proceed`, or `llm` (ask a bigger model).
6. If the Decider server is down, `on_decider_error` decides (default `deny`, fail closed). A circuit breaker stops a dead server from slowing every call.

What counts as evidence: `user` messages and `tool_result` data. The agent's own earlier messages do **not** count, so it cannot approve itself.

## Roll out safely: shadow mode, then enforce

```yaml
defaults: { mode: shadow }   # log what would happen, change nothing (hard rules still enforce)
```
```bash
deciderkit summarize audit.jsonl     # what would shadow mode have blocked?
```
Flip to `mode: enforce` when the log looks right.

## The confidence ladder

Decider handles the easy cases quickly. Only the uncertain middle band is passed to a bigger model:

```python
from deciderkit.escalate import BedrockEscalator
guard = DeciderGuard("recipe.yaml", escalator=BedrockEscalator("<bedrock model id>"))
# recipe: defaults: { uncertain: llm }
```
Any callable works as an escalator: `fn(ctx) -> True | False | None`.

## Tune thresholds from data

The default thresholds (0.80 and 0.30) are starting points, not tuned values. Label some audit rows (`"label": true` when the call really was fine), then:

```bash
deciderkit tune benchmarks/results/<run>-rows.jsonl --check grounded_args
```
It prints `allow_above` and `block_below` lines that meet your precision targets, and refuses when the data is too noisy.

## Benchmark

`benchmarks/cases.jsonl` has 100 support-ops cases (50 fine, 50 not), made by `benchmarks/make_dataset.py`. Labels come from the generator templates, not from human review, so read a sample and add cases from your own domain.

```bash
python benchmarks/run_benchmark.py --mock      # harness check only, numbers are NOT real
python benchmarks/run_benchmark.py --decider-url http://localhost:8099 --llm-model-id <id> ...
python benchmarks/make_dataset.py --seed 7 --out benchmarks/cases_holdout.jsonl   # fresh hold-out set
```
It compares Decider alone, an LLM alone, and the ladder on accuracy, bad calls let through, good calls blocked, latency and LLM cost.

**Results** (real run: Decider checkpoint `StrandsAgents/strands-decider-2B-hobson-v19` on a g5.xlarge with an A10G GPU; second-opinion LLM: Amazon Nova 2 Lite via `us.amazon.nova-2-lite-v1:0`; list prices $0.33 in / $2.75 out per 1M tokens, taken from a third-party price page, so verify them).

Thresholds were tuned with `deciderkit tune` on `cases.jsonl` (100 cases, seed 1812) and then tested on a fresh 100-case hold-out set (`--seed 7`).

| Hold-out, tuned recipe | Accuracy | Bad calls let through | Good calls blocked | p50 ms | p95 ms | LLM calls | LLM cost |
|---|---|---|---|---|---|---|---|
| Decider alone | 76% | 1 | 23 | 180 | 186 | 0 | $0 |
| LLM alone (Nova 2 Lite) | 100% | 0 | 0 | 2763 | 5778 | 100 | $0.0203 |
| **Ladder** (Decider, then LLM only if unsure) | 99% | 1 | 0 | **817** | 4808 | **57** | **$0.0115** |

What this shows: the ladder cut median latency about 3.4x and LLM calls and LLM cost by about 43%, in exchange for 1 bad call let through out of 100 (the LLM alone let through none).

What it does not show: each set is only 100 synthetic cases whose labels come from templates, each number is from one run, and one extra error is not statistically meaningful. With the untuned default thresholds (0.80 / 0.30) the ladder gave no real benefit (hold-out p50 2948 ms vs 2516 ms for the LLM alone, 93 LLM calls), and Decider alone blocked 43 good calls. Tuning thresholds on your own data matters.

## Honest limits

- Decider 2B is experimental and can be wrong. Keep hard rules for anything that must never happen.
- `grounded_args` judges all arguments together; it does not tell you which one was guessed.
- The transcript sent to Decider is capped (about 1800 characters) because the Decider training tools truncate states to about 2000 characters by default. Long conversations lose their oldest messages.
- `Confirm` pauses the agent and needs your app to handle Strands interrupts; the demo uses `guide` instead.
- Tests do not cover live Bedrock or the real Decider model. Run the real benchmark and one end-to-end demo before you rely on it.
- Do not report numbers measured against the mock server.

## Layout

```
deciderkit/   client, recipe, engine, guard (Strands adapter), audit, tune, escalate, cli
recipes/      support_ops.yaml
examples/     support_agent.py
benchmarks/   cases.jsonl, make_dataset.py, run_benchmark.py
tests/        pytest suite (mock server + real Strands Agent with a scripted model)
docs/         aws-setup.md, article-outline.md, video-script.md, submission-checklist.md
```

## License

MIT
