# Demo video script (target 2:30, hard limit under 3:00)

| Time | Screen | Say (short) |
|---|---|---|
| 0:00-0:15 | Title + one sentence | "Agents guess missing details. When the tool moves money, that is a real problem." |
| 0:15-0:50 | `--no-guard` run: "refund my order" | Agent invents an order and refunds it. Show `Refunds issued: ['ORD-...']`. |
| 0:50-1:25 | Guarded run, same prompt | Guard prints ASK. The agent asks "which order?". Show the `[guard]` line with the P(true). |
| 1:25-1:50 | Grounded prompt (full amount of ORD-482913) | Guard prints ALLOW and the refund runs. |
| 1:50-2:10 | The 10-line `support_ops.yaml` + the one line of Python | "Any Strands developer can reuse this by editing YAML." |
| 2:10-2:35 | Benchmark table + `deciderkit summarize` | Real numbers: accuracy, bad calls let through, latency, cost. |
| 2:35-2:50 | Repo link + tags | "Experimental, MIT licensed. Link below." |

Record checklist: run `python examples/support_agent.py --no-guard --demo` and `--demo` (guarded) against the REAL Decider server, not the mock. Show your hardware. Do not claim a latency you did not measure.
