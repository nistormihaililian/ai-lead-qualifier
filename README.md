1. AI Lead Qualifier
Automated lead qualification system that reads leads from a CSV file, uses Claude (Anthropic API) to analyze each lead's context and decide whether it's sales-ready or needs nurturing, then writes the enriched results to a new CSV.

Built as a foundation for larger agent-based automation systems.

2. What it does

- Reads leads from `leads.csv` (name, email, score, source)
- Sends each lead to Claude for AI-based qualification (not a fixed score threshold — the model reasons about score and source together)
- Parses the AI's decision and reasoning
- Writes results to `leads_processed.csv` with two new columns: `ai_decision`, `ai_reason`
- Handles API failures gracefully without stopping the batch

3. Example

**Input (`leads.csv`):**

| name | email | score  | source |
| Ion Popescu | ion@firma.ro | 85 | Facebook Ads |
| Andrei Stan | andrei@firma.ro | 92 | Referral |

**Output (`leads_processed.csv`):**

| name | ... | ai_decision | ai_reason |
| Ion Popescu | ... | Qualified | High score indicates strong purchase intent |
| Andrei Stan | ... | Qualified | High score, referral source indicates strong intent |

4. Tech stack

- Python 3.12
- Anthropic API (Claude Haiku 4.5)
- CSV processing (standard library)

5. Setup

1. Clone this repo
2. Create a virtual environment and install dependencies:
------
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
------
6. Copy `.env.example` to `.env` and add your Anthropic API key:
ANTHROPIC_API_KEY=your_key_here
7. Add your leads to `leads.csv` (same columns as the example)
8. Run:
oython main.py

########## Roadmap ###########

This project is the first step in a larger multi-agent automation system. Planned additions:
- LangGraph-based multi-step agent workflow
- Automated email follow-up for qualified leads
- Database persistence (Supabase/PostgreSQL)
- API endpoint (FastAPI) for real-time lead scoring