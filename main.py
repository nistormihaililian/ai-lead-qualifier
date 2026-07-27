import csv
import os
import time
import logging
import argparse
from typing import TypedDict
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)


class LeadState(TypedDict):
    name: str
    email: str
    score: str
    source: str
    raw_response: str
    decision: str
    reason: str
    attempt: int
    failed: bool


def call_claude(state: LeadState) -> LeadState:
    state["attempt"] += 1
    logging.info(f"Attempt {state['attempt']}/{MAX_RETRIES} for {state['name']}")

    prompt = f"""You are a sales lead qualifier.

Lead info:
Name: {state['name']}
Email: {state['email']}
Score: {state['score']}
Source: {state['source']}

Decide if this lead is qualified for sales follow-up or should go to a nurture sequence.
Reply with ONLY one line in this exact format:
DECISION: <Qualified or Nurture> | REASON: <short reason, max 10 words>
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        state["raw_response"] = response.content[0].text.strip()
        state["failed"] = False
    except Exception as error:
        logging.warning(f"API call failed for {state['name']}: {error}")
        state["raw_response"] = ""
        state["failed"] = True

    return state


def route_after_call(state: LeadState) -> str:
    if not state["failed"]:
        return "parse"
    if state["attempt"] < MAX_RETRIES:
        return "retry"
    return "give_up"


def wait_and_retry(state: LeadState) -> LeadState:
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def parse_response(state: LeadState) -> LeadState:
    try:
        decision_part = state["raw_response"].split("|")[0]
        reason_part = state["raw_response"].split("|")[1]
        state["decision"] = decision_part.replace("DECISION:", "").strip()
        state["reason"] = reason_part.replace("REASON:", "").strip()
        logging.info(f"Processed {state['name']} -> {state['decision']} | {state['reason']}")
    except Exception as error:
        logging.warning(f"Failed to parse response for {state['name']}: {error}")
        state["decision"] = "Error"
        state["reason"] = "Could not parse AI response"
    return state


def mark_failed(state: LeadState) -> LeadState:
    logging.error(f"Giving up on {state['name']} after {state['attempt']} attempts")
    state["decision"] = "Error"
    state["reason"] = "Failed after multiple retries"
    return state


graph = StateGraph(LeadState)

graph.add_node("call_claude", call_claude)
graph.add_node("wait_and_retry", wait_and_retry)
graph.add_node("parse_response", parse_response)
graph.add_node("mark_failed", mark_failed)

graph.set_entry_point("call_claude")

graph.add_conditional_edges(
    "call_claude",
    route_after_call,
    {
        "parse": "parse_response",
        "retry": "wait_and_retry",
        "give_up": "mark_failed"
    }
)

graph.add_edge("wait_and_retry", "call_claude")
graph.add_edge("parse_response", END)
graph.add_edge("mark_failed", END)

lead_graph = graph.compile()


def parse_args():
    parser = argparse.ArgumentParser(description="AI Lead Qualifier - qualify sales leads using Claude + LangGraph")
    parser.add_argument("--input", default="leads.csv", help="Path to the input CSV file (default: leads.csv)")
    parser.add_argument("--output", default="leads_processed.csv", help="Path to the output CSV file (default: leads_processed.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.info("Starting AI Lead Qualifier run (LangGraph)")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")

    with open(args.input, mode="r") as file:
        reader = csv.DictReader(file)
        leads = list(reader)

    logging.info(f"Loaded {len(leads)} leads from {args.input}")

    processed_leads = []

    for lead in leads:
        initial_state: LeadState = {
            "name": lead["name"],
            "email": lead["email"],
            "score": lead["score"],
            "source": lead["source"],
            "raw_response": "",
            "decision": "",
            "reason": "",
            "attempt": 0,
            "failed": False
        }

        final_state = lead_graph.invoke(initial_state)

        lead["ai_decision"] = final_state["decision"]
        lead["ai_reason"] = final_state["reason"]
        processed_leads.append(lead)

    fieldnames = list(processed_leads[0].keys())

    with open(args.output, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_leads)

    logging.info(f"Run complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()