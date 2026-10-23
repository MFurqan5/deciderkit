# Submission checklist (deadline: Oct 23, 11:59 PM PDT = Oct 24, 11:59 AM in Lahore. Aim for Oct 22.)

Rules on the hackathon page (re-read the Rules tab for anything I could not see):
- [ ] Agent uses Decider 2B for at least one meaningful decision, paired with Amazon Bedrock
- [ ] Article on Builder Center (agent, what it does, the decision handed to Decider)
- [ ] Public repo with source and at least one reusable decision pattern (typed question + options, wiring, usage note)
- [ ] Demo video under 3 minutes showing the agent and the Decider decision working
- [ ] One pattern declared: `#guardrails`
- [ ] Tags added: `#strands-decider-2b` and `#guardrails`
- [ ] Original, not published before: keep the repo PRIVATE until submission day

Before you press submit:
- [ ] `pytest -q` passes; `deciderkit validate recipes/support_ops.yaml` passes
- [ ] One real end-to-end run: real Decider server + real Bedrock (the tests do not cover this)
- [ ] Real benchmark run saved in `benchmarks/results/`; table pasted into README and article
- [ ] README "Results" section filled with real numbers, hardware and checkpoint
- [ ] Read 10 random lines of `benchmarks/cases.jsonl` to check the labels make sense
- [ ] Searched GitHub / PyPI / Builder Center again for a similar package
- [ ] Removed `audit.jsonl`, keys and personal data from the repo
- [ ] Stopped and terminated the AWS GPU instance
- [ ] Repo made public; links in the article work in a private browser window
