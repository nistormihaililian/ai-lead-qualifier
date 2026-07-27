# AI Lead Qualifier

A lead qualification pipeline built as a **LangGraph state machine**. Each lead flows through a graph of nodes — API call, retry routing, response parsing — instead of a linear script with error handling bolted on.

Reads leads from a CSV, has Claude reason about each one's context (score + source, not a fixed threshold), and writes back an enriched CSV with the AI's decision and reasoning.

## Why LangGraph, not just a script

Retry logic, in most scripts, is a `try/except` wrapped in a `for` loop — buried inside a function, invisible until you read the code. Here it's a first-class part of the graph: a failed API call routes to a `wait_and_retry` node, which loops back to `call_claude`, up to a configurable retry limit, before falling through to a `mark_failed` node.

That structure is the whole point of using a graph over a script: the control flow is explicit, inspectable, and easy to extend — add a human-review node, a second-opinion node, or a Slack-notification node without restructuring the file.

## How it works
call_claude ──success──> parse_response ──> END
│
├──fail, retries left──> wait_and_retry ──> call_claude (loop)
│
└──fail, out of retries──> mark_failed ──> END
1. Reads leads from `leads.csv`
2. For each lead, runs the graph above via LangGraph
3. Claude (Haiku 4.5) reasons about the lead's score and source together
4. Parses the decision and reasoning from the response
5. Writes results to `leads_processed.csv`, original columns plus `ai_decision` and `ai_reason`
6. Every step is logged to both console and `run.log`, with timestamps

## Example

**Input**

| name | email | score | source |
|---|---|---|---|
| Ion Popescu | ion@firma.ro | 85 | Facebook Ads |
| Maria Ionescu | maria@firma.ro | 45 | Google Ads |
| Andrei Stan | andrei@firma.ro | 92 | Referral |

**Output**

| name | ai_decision | ai_reason |
|---|---|---|
| Ion Popescu | Qualified | High score indicates strong purchase intent |
| Maria Ionescu | Nurture | Low score, needs engagement first |
| Andrei Stan | Qualified | High score and referral source indicate strong fit |

## Stack

- Python 3.12
- LangGraph — state machine / graph orchestration
- Anthropic API — Claude Haiku 4.5
- `argparse` — CLI interface
- `logging` — structured, timestamped logs to file and console
- `csv` (standard library)

## Setup

```bash
git clone https://github.com/iuliantiu/ai-lead-qualifier.git
cd ai-lead-qualifier

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:
ANTHROPIC_API_KEY=your_key_here
Run with the default files:

```bash
python main.py
```

Or point it at any CSV with the same columns (`name, email, score, source`):

```bash
python main.py --input other_leads.csv --output results.csv
```

## Design notes

- **Graph-based retry**: failed API calls route through a dedicated retry node with a configurable delay and attempt limit, rather than a hidden loop inside a function.
- **Typed state**: the state passed through the graph is a `TypedDict`, so every node knows exactly what fields it can read and write.
- **Failure isolation**: a lead that exhausts all retries is marked `Error` in the output instead of crashing the batch.
- **Model choice**: Haiku handles this classification task well and costs a fraction of Sonnet.
- **No hardcoded thresholds**: qualification logic lives in the prompt, not in an `if` statement — tuning the criteria doesn't require touching the code.

## Roadmap

- [x] LangGraph state machine with retry routing
- [ ] Multi-agent workflow with CrewAI (e.g. a second agent drafting follow-up copy for qualified leads)
- [ ] Persistent storage (Supabase/PostgreSQL) instead of CSV in/out
- [ ] FastAPI endpoint for real-time scoring, callable from a webhook (n8n, GHL, etc.) with a live demo link

## License

MIT