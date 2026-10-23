# AWS setup (run Decider on a GPU, keep $100 safe)

I could not test these steps on a live AWS box. Treat them as a checklist and fix what differs in your account.

## 0. Before you spend anything
1. Billing alerts at $25, $50, $75 (Billing > Budgets).
2. Request GPU quota: Service Quotas > EC2 > "Running On-Demand G and VT instances". New accounts are often at 0. Approval can take time, so do this first.
3. Bedrock model access for the agent's LLM (Bedrock console > Model access). Use a cheap model for the benchmark.

## 1. Launch the GPU server
- Instance: `g5.xlarge` (1x A10G). About $1/hour on-demand (check the current price in your region).
- AMI: an Ubuntu GPU / Deep Learning AMI that already has NVIDIA drivers.
- Disk: 50 GB is enough.
- Security group: **allow SSH (22) from your IP only. Do NOT open port 8099.** Assume the Decider server has no login.

## 2. Install and start Decider (on the server)
```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
```
- The server listens on `127.0.0.1` by default. That is what we want.
- The first start downloads the model from Hugging Face. Wait for it to finish.
- Newer checkpoints exist. Check the official repo (`strands-labs/strands-decider`) for the one it recommends, and note which one you used in your article.
- Test on the server: `curl http://127.0.0.1:8099/health`

## 3. Connect from your laptop (SSH tunnel, nothing exposed)
```bash
ssh -i your-key.pem -L 8099:127.0.0.1:8099 ubuntu@<server-public-ip>
```
Leave that window open. On your laptop, in another window:
```bash
export STRANDS_DECIDER_URL=http://localhost:8099
deciderkit health
```

## 4. Run the benchmark (this is where the GPU hours go)
```bash
python benchmarks/run_benchmark.py --decider-url http://localhost:8099 \
  --llm-model-id <your bedrock model id> --region <region> \
  --llm-input-price <USD per 1M input tokens> --llm-output-price <USD per 1M output tokens> \
  --hardware "g5.xlarge (A10G)"
```
Look up the token prices yourself on the Bedrock pricing page. Do not guess them.

## 5. STOP THE SERVER
Stop (not just disconnect) the instance after every session. At about $1/hour, a forgotten night costs about $24.
When the hackathon is over, terminate the instance, delete the disk and release any Elastic IP.

## Budget (rough, my estimate)
GPU 10 to 20 hours = $10 to $20 | Bedrock calls for 100 cases x 3 setups = a few dollars | extras = a few dollars.
