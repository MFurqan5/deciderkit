# Article outline (Builder Center, tags: #strands-decider-2b #guardrails)

Write it in your own voice. Fill every [bracket] with a real result from YOUR run. Do not invent numbers.

**Title idea:** Stop your agent from guessing: decision recipes for Strands Decider 2B

1. **The problem (short).** An agent refunds the wrong order because the user never gave an order ID. Real cost, no human in the loop.
2. **The decision I gave Decider 2B.** One typed question before every risky tool call: "Is every argument stated by the user or a tool result?" Pattern: `#guardrails`.
3. **The agent.** Support-ops agent on Bedrock (model: [model id]) with lookup, refund, cancel and update-address tools.
4. **The reusable part: DeciderKit.**
   - The recipe file (paste the 10-line YAML).
   - The one-line wiring (`interventions=[DeciderGuard(...)]`).
   - Why hard rules run first and Decider handles only the fuzzy part.
5. **What makes it safe to adopt.** Shadow mode, fail-closed on outage, circuit breaker, audit log, evidence rule (the agent's own words are not evidence).
6. **The confidence ladder and tuning.** Easy cases decided fast, the uncertain band escalated, thresholds tuned from labeled logs instead of by hand.
7. **Results (real numbers only).** Table from `benchmarks/results/*.md`: accuracy, bad calls let through, good calls blocked, p50/p95 latency, LLM calls and cost. State the hardware ([g5.xlarge A10G]) and the checkpoint ([hobson-v19]).
8. **What I learned / what failed.** Include at least two real findings, for example which categories were hard (`wrong_order_from_lookup`, `not_requested`?) and what you changed. Say plainly that Decider 2B is experimental.
9. **Try it.** Repo link, install, recipe copy-paste, how to add your own tool in 2 minutes.

Judges score: creativity 30%, effective use of Decider 30%, communication 25%, originality 15%. Show the failure and the fix, not only the success.
