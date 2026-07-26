import csv
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

        return decision, reason

    except Exception as error:
        print(f"[ERROR] Failed to process lead {lead['name']}: {error}")
        return "Error", "Could not process lead"

with open("leads.csv", mode="r") as file:
    reader = csv.DictReader(file)
    leads = list(reader)

processed_leads = []

for lead in leads:
    decision, reason = qualify_lead(lead)
    lead["ai_decision"] = decision
    lead["ai_reason"] = reason
    processed_leads.append(lead)
    print(lead["name"], "->", decision, "|", reason)

fieldnames = list(processed_leads[0].keys())

with open("leads_processed.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(processed_leads)

print("\nDone. Results saved to leads_processed.csv")