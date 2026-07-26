import csv
import os
import time
import logging
import argparse
from dotenv import load_dotenv
from anthropic import Anthropic

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


def qualify_lead(lead):
    prompt = f"""You are a sales lead qualifier.

Lead info:
Name: {lead['name']}
Email: {lead['email']}
Score: {lead['score']}
Source: {lead['source']}

Decide if this lead is qualified for sales follow-up or should go to a nurture sequence.
Reply with ONLY one line in this exact format:
DECISION: <Qualified or Nurture> | REASON: <short reason, max 10 words>
"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            raw_text = response.content[0].text.strip()

            decision_part = raw_text.split("|")[0]
            reason_part = raw_text.split("|")[1]

            decision = decision_part.replace("DECISION:", "").strip()
            reason = reason_part.replace("REASON:", "").strip()

            logging.info(f"Processed {lead['name']} -> {decision} | {reason}")
            return decision, reason

        except Exception as error:
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {lead['name']}: {error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    logging.error(f"Giving up on {lead['name']} after {MAX_RETRIES} attempts")
    return "Error", "Failed after multiple retries"


def parse_args():
    parser = argparse.ArgumentParser(description="AI Lead Qualifier - qualify sales leads using Claude")
    parser.add_argument(
        "--input",
        default="leads.csv",
        help="Path to the input CSV file (default: leads.csv)"
    )
    parser.add_argument(
        "--output",
        default="leads_processed.csv",
        help="Path to the output CSV file (default: leads_processed.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.info("Starting AI Lead Qualifier run")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")

    with open(args.input, mode="r") as file:
        reader = csv.DictReader(file)
        leads = list(reader)

    logging.info(f"Loaded {len(leads)} leads from {args.input}")

    processed_leads = []

    for lead in leads:
        decision, reason = qualify_lead(lead)
        lead["ai_decision"] = decision
        lead["ai_reason"] = reason
        processed_leads.append(lead)

    fieldnames = list(processed_leads[0].keys())

    with open(args.output, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_leads)

    logging.info(f"Run complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()