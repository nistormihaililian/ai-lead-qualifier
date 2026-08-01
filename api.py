from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import time
import logging
from typing import TypedDict
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from supabase import create_client
from crewai import Agent, Task, Crew
import resend

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

resend.api_key = os.getenv("RESEND_API_KEY")

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
    source: str
    message: str
    raw_score_response: str
    score: int
    score_reason: str
    score_attempt: int
    score_failed: bool
    raw_response: str
    decision: str
    reason: str
    attempt: int
    failed: bool


def score_lead(state: LeadState) -> LeadState:
    state["score_attempt"] += 1
    logging.info(f"Scoring attempt {state['score_attempt']}/{MAX_RETRIES} for {state['name']}")

    prompt = f"""You are a sales lead scoring assistant.

A lead sent this message:
Name: {state['name']}
Source: {state['source']}
Message: "{state['message']}"

Analyze the message for buying intent, urgency, and budget signals.
Reply with ONLY one line in this exact format:
SCORE: <a number from 0 to 100> | REASON: <short reason, max 12 words>
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        state["raw_score_response"] = response.content[0].text.strip()
        state["score_failed"] = False
    except Exception as error:
        logging.warning(f"Scoring API call failed for {state['name']}: {error}")
        state["raw_score_response"] = ""
        state["score_failed"] = True

    return state


def route_after_score(state: LeadState) -> str:
    if not state["score_failed"]:
        return "parse_score"
    if state["score_attempt"] < MAX_RETRIES:
        return "retry_score"
    return "give_up_score"


def wait_and_retry_score(state: LeadState) -> LeadState:
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def parse_score(state: LeadState) -> LeadState:
    try:
        score_part = state["raw_score_response"].split("|")[0]
        reason_part = state["raw_score_response"].split("|")[1]
        score_text = score_part.replace("SCORE:", "").strip()
        state["score"] = int(score_text)
        state["score_reason"] = reason_part.replace("REASON:", "").strip()
        logging.info(f"Scored {state['name']} -> {state['score']} | {state['score_reason']}")
    except Exception as error:
        logging.warning(f"Failed to parse score for {state['name']}: {error}")
        state["score"] = 0
        state["score_reason"] = "Could not parse AI scoring response"
    return state


def mark_score_failed(state: LeadState) -> LeadState:
    logging.error(f"Giving up on scoring {state['name']} after {state['score_attempt']} attempts")
    state["score"] = 0
    state["score_reason"] = "Scoring failed after multiple retries"
    return state


def call_claude(state: LeadState) -> LeadState:
    state["attempt"] += 1
    logging.info(f"Qualification attempt {state['attempt']}/{MAX_RETRIES} for {state['name']}")

    prompt = f"""You are a sales lead qualifier.

Lead info:
Name: {state['name']}
Message: "{state['message']}"
Score: {state['score']} (reason: {state['score_reason']})
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
graph.add_node("score_lead", score_lead)
graph.add_node("wait_and_retry_score", wait_and_retry_score)
graph.add_node("parse_score", parse_score)
graph.add_node("mark_score_failed", mark_score_failed)
graph.add_node("call_claude", call_claude)
graph.add_node("wait_and_retry", wait_and_retry)
graph.add_node("parse_response", parse_response)
graph.add_node("mark_failed", mark_failed)

graph.set_entry_point("score_lead")

graph.add_conditional_edges(
    "score_lead",
    route_after_score,
    {"parse_score": "parse_score", "retry_score": "wait_and_retry_score", "give_up_score": "mark_score_failed"}
)
graph.add_edge("wait_and_retry_score", "score_lead")
graph.add_edge("parse_score", "call_claude")
graph.add_edge("mark_score_failed", "call_claude")

graph.add_conditional_edges(
    "call_claude",
    route_after_call,
    {"parse": "parse_response", "retry": "wait_and_retry", "give_up": "mark_failed"}
)
graph.add_edge("wait_and_retry", "call_claude")
graph.add_edge("parse_response", END)
graph.add_edge("mark_failed", END)

lead_graph = graph.compile()


def write_followup_email(name, source, reason):
    copywriter = Agent(
        role="Sales Follow-up Copywriter",
        goal="Write short, personalized follow-up messages for qualified sales leads",
        backstory="You are an experienced sales copywriter who writes concise, "
                   "friendly follow-up emails that get replies, without sounding pushy.",
        llm="anthropic/claude-haiku-4-5-20251001",
        verbose=False
    )

    task = Task(
        description=f"Write a short follow-up email (max 60 words) for this lead: "
                     f"Name: {name}, Source: {source}, Qualification reason: {reason}. "
                     f"Keep it warm, direct, and mention the source naturally.",
        expected_output="Only the email itself: a subject line followed by the email body. "
                         "No word count, no notes, no explanations before or after the email.",
        agent=copywriter
    )

    crew = Crew(agents=[copywriter], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


def send_followup_email(to_email, name, email_content):
    try:
        lines = email_content.strip().split("\n", 1)
        subject = lines[0].replace("Subject:", "").strip() if len(lines) > 1 else f"Following up, {name}"
        body = lines[1].strip() if len(lines) > 1 else email_content

        resend.Emails.send({
            "from": "AI Lead Qualifier <onboarding@resend.dev>",
            "to": [to_email],
            "subject": subject,
            "text": body
        })
        logging.info(f"Follow-up email sent to {to_email}")
        return True
    except Exception as error:
        logging.warning(f"Failed to send email to {to_email}: {error}")
        return False


app = FastAPI(title="AI Lead Qualifier API")


class LeadInput(BaseModel):
    name: str
    email: str
    source: str
    message: str


class LeadOutput(BaseModel):
    name: str
    score: int
    score_reason: str
    decision: str
    reason: str
    followup_email: str = ""
    email_sent: bool = False


@app.get("/")
def read_root():
    return {"message": "AI Lead Qualifier API is running"}


@app.post("/qualify", response_model=LeadOutput)
def qualify_lead(lead: LeadInput):
    initial_state: LeadState = {
        "name": lead.name,
        "email": lead.email,
        "source": lead.source,
        "message": lead.message,
        "raw_score_response": "",
        "score": 0,
        "score_reason": "",
        "score_attempt": 0,
        "score_failed": False,
        "raw_response": "",
        "decision": "",
        "reason": "",
        "attempt": 0,
        "failed": False
    }

    final_state = lead_graph.invoke(initial_state)

    followup_email = ""
    email_sent = False
    if final_state["decision"] == "Qualified":
        logging.info(f"Generating follow-up email for {lead.name}")
        followup_email = write_followup_email(lead.name, lead.source, final_state["reason"])
        email_sent = send_followup_email(lead.email, lead.name, followup_email)

    supabase.table("leads").insert({
        "name": lead.name,
        "email": lead.email,
        "score": final_state["score"],
        "source": lead.source,
        "ai_decision": final_state["decision"],
        "ai_reason": final_state["reason"],
        "followup_email": followup_email if followup_email else None,
        "email_sent": email_sent
    }).execute()

    logging.info(f"Saved {lead.name} to Supabase")

    return LeadOutput(
        name=lead.name,
        score=final_state["score"],
        score_reason=final_state["score_reason"],
        decision=final_state["decision"],
        reason=final_state["reason"],
        followup_email=followup_email,
        email_sent=email_sent
    )


@app.get("/form", response_class=HTMLResponse)
def serve_form():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Lead Qualifier</title>
<style>
    body {
        font-family: Arial, sans-serif;
        max-width: 480px;
        margin: 60px auto;
        padding: 20px;
        background: #f5f5f5;
    }
    h1 {
        font-size: 22px;
        margin-bottom: 20px;
    }
    label {
        display: block;
        margin-top: 14px;
        font-weight: bold;
        font-size: 14px;
    }
    input, textarea {
        width: 100%;
        padding: 8px;
        margin-top: 4px;
        box-sizing: border-box;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-family: inherit;
    }
    textarea {
        min-height: 80px;
        resize: vertical;
    }
    button {
        margin-top: 20px;
        padding: 10px 16px;
        background: #222;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 15px;
    }
    button:disabled {
        background: #999;
    }
    #result {
        margin-top: 24px;
        padding: 16px;
        border-radius: 6px;
        display: none;
    }
    #result.qualified {
        background: #e6f4ea;
        border: 1px solid #34a853;
    }
    #result.nurture {
        background: #fef7e0;
        border: 1px solid #f9ab00;
    }
    #result.error {
        background: #fce8e6;
        border: 1px solid #ea4335;
    }
</style>
</head>
<body>

<h1>AI Lead Qualifier</h1>

<form id="leadForm">
    <label for="name">Name</label>
    <input type="text" id="name" required>

    <label for="email">Email</label>
    <input type="email" id="email" required>

    <label for="source">Source</label>
    <input type="text" id="source" placeholder="e.g. Facebook Ads" required>

    <label for="message">Message</label>
    <textarea id="message" placeholder="What is the lead interested in? Any budget or timeline mentioned?" required></textarea>

    <button type="submit" id="submitBtn">Submit</button>
</form>

<div id="result"></div>

<script>
const form = document.getElementById("leadForm");
const resultDiv = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    submitBtn.disabled = true;
    submitBtn.textContent = "Processing...";
    resultDiv.style.display = "none";

    const payload = {
        name: document.getElementById("name").value,
        email: document.getElementById("email").value,
        source: document.getElementById("source").value,
        message: document.getElementById("message").value
    };

    try {
        const response = await fetch("/qualify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        resultDiv.style.display = "block";
        resultDiv.className = "";

        if (data.decision === "Qualified") {
            resultDiv.classList.add("qualified");
        } else if (data.decision === "Nurture") {
            resultDiv.classList.add("nurture");
        } else {
            resultDiv.classList.add("error");
        }

        let html = "<strong>Score:</strong> " + data.score + " (" + data.score_reason + ")<br>";
        html += "<strong>Decision:</strong> " + data.decision + "<br>";
        html += "<strong>Reason:</strong> " + data.reason;

        if (data.followup_email) {
            html += "<br><br><strong>Follow-up email:</strong><br>" +
                    data.followup_email.replace(/\\n/g, "<br>");
            html += "<br><br><strong>Email sent:</strong> " + (data.email_sent ? "Yes" : "No");
        }

        resultDiv.innerHTML = html;

    } catch (error) {
        resultDiv.style.display = "block";
        resultDiv.className = "error";
        resultDiv.textContent = "Something went wrong. Check the server logs.";
    }

    submitBtn.disabled = false;
    submitBtn.textContent = "Submit";
});
</script>

</body>
</html>
"""