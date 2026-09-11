from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import time
import logging
import requests
from datetime import datetime, timezone
from typing import TypedDict
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from supabase import create_client
from crewai import Agent, Task, Crew
import resend
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

resend.api_key = os.getenv("RESEND_API_KEY")

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
MAX_SUBMISSIONS_PER_IP = 3

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
    email_type: str
    email_confidence: int
    email_reason: str
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


def analyze_email_domain(email, name, message):
    analyst = Agent(
        role="Lead Verification & Domain Intelligence Analyst",
        goal="Assess whether a lead's email address signals genuine business intent "
             "or is a low-commitment personal or disposable address",
        backstory="You are an experienced lead-quality analyst who has reviewed thousands of B2B leads. "
                   "You know which providers are personal (gmail, yahoo, hotmail, outlook, icloud, proton), "
                   "which patterns suggest disposable or temporary emails (tempmail, mailinator, "
                   "guerrillamail, 10minutemail, random character strings, throwaway domains), "
                   "and which patterns look like real company domains.",
        llm="anthropic/claude-haiku-4-5-20251001",
        verbose=False
    )

    task = Task(
        description=f"Analyze this lead's email address for business legitimacy signals.\n\n"
                     f"Name: {name}\n"
                     f"Email: {email}\n"
                     f"Message context: \"{message}\"\n\n"
                     f"Classify the email and explain your reasoning briefly.",
        expected_output="Only one line in this exact format, nothing else: "
                         "TYPE: <business or personal or disposable> | CONFIDENCE: <0-100> | REASON: <short reason, max 12 words>",
        agent=analyst
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


def parse_email_analysis(raw_text):
    try:
        parts = raw_text.strip().split("|")
        email_type = parts[0].replace("TYPE:", "").strip().lower()
        confidence = int(parts[1].replace("CONFIDENCE:", "").strip())
        reason = parts[2].replace("REASON:", "").strip()
        return email_type, confidence, reason
    except Exception as error:
        logging.warning(f"Failed to parse email analysis: {error}")
        return "unknown", 0, "Could not parse email domain analysis"


def score_lead(state: LeadState) -> LeadState:
    state["score_attempt"] += 1
    logging.info(f"Scoring attempt {state['score_attempt']}/{MAX_RETRIES} for {state['name']}")

    prompt = f"""You are a sales lead scoring assistant.

A lead sent this message:
Name: {state['name']}
Source: {state['source']}
Message: "{state['message']}"

Email domain analysis (from a separate verification agent):
Type: {state['email_type']} (confidence: {state['email_confidence']})
Reason: {state['email_reason']}

Analyze the message for buying intent, urgency, and budget signals.
Factor in the email domain analysis: a business email is a stronger signal of serious intent,
a disposable email is a red flag that should lower the score significantly, a personal email
(gmail, yahoo, etc.) is neutral and common for individual consumers.

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
Email type: {state['email_type']} (reason: {state['email_reason']})
Source: {state['source']}

Decide if this lead is qualified for sales follow-up or should go to a nurture sequence.
A disposable email is a strong signal against qualifying, regardless of message content.
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


def send_slack_notification(name, email, score, source, reason):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("SLACK_WEBHOOK_URL not set, skipping notification")
        return False

    message = {
        "text": f"New Qualified Lead: {name}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*New Qualified Lead*\n"
                            f"*Name:* {name}\n"
                            f"*Email:* {email}\n"
                            f"*Score:* {score}\n"
                            f"*Source:* {source}\n"
                            f"*Reason:* {reason}"
                }
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=message, timeout=5)
        response.raise_for_status()
        logging.info(f"Slack notification sent for {name}")
        return True
    except Exception as error:
        logging.warning(f"Failed to send Slack notification for {name}: {error}")
        return False


def send_outbound_webhook(lead_output: dict):
    webhook_url = os.getenv("OUTBOUND_WEBHOOK_URL")
    if not webhook_url:
        return False

    try:
        response = requests.post(webhook_url, json=lead_output, timeout=5)
        response.raise_for_status()
        logging.info(f"Outbound webhook sent for {lead_output.get('name')}")
        return True
    except Exception as error:
        logging.warning(f"Failed to send outbound webhook: {error}")
        return False


def verify_api_key(x_api_key: str = Header(...)):
    expected_key = os.getenv("API_SECRET_KEY")
    if x_api_key != expected_key:
        logging.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def check_and_increment_ip_usage(ip_address: str) -> bool:
    result = supabase.table("ip_usage").select("submission_count").eq("ip_address", ip_address).execute()

    if result.data:
        count = result.data[0]["submission_count"]
        if count >= MAX_SUBMISSIONS_PER_IP:
            return False
        supabase.table("ip_usage").update({
            "submission_count": count + 1,
            "last_submission": datetime.now(timezone.utc).isoformat()
        }).eq("ip_address", ip_address).execute()
    else:
        supabase.table("ip_usage").insert({
            "ip_address": ip_address,
            "submission_count": 1
        }).execute()

    return True


app = FastAPI(title="AI Lead Qualifier API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
    email_type: str = ""
    email_confidence: int = 0
    email_reason: str = ""
    followup_email: str = ""
    email_sent: bool = False


def process_lead(lead: LeadInput) -> LeadOutput:
    logging.info(f"Analyzing email domain for {lead.name}")
    raw_email_analysis = analyze_email_domain(lead.email, lead.name, lead.message)
    email_type, email_confidence, email_reason = parse_email_analysis(raw_email_analysis)
    logging.info(f"Email analysis for {lead.name} -> {email_type} ({email_confidence}) | {email_reason}")

    initial_state: LeadState = {
        "name": lead.name,
        "email": lead.email,
        "source": lead.source,
        "message": lead.message,
        "email_type": email_type,
        "email_confidence": email_confidence,
        "email_reason": email_reason,
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
        send_slack_notification(lead.name, lead.email, final_state["score"], lead.source, final_state["reason"])

    supabase.table("leads").insert({
        "name": lead.name,
        "email": lead.email,
        "score": final_state["score"],
        "source": lead.source,
        "ai_decision": final_state["decision"],
        "ai_reason": final_state["reason"],
        "email_type": email_type,
        "email_confidence": email_confidence,
        "email_reason": email_reason,
        "followup_email": followup_email if followup_email else None,
        "email_sent": email_sent
    }).execute()

    logging.info(f"Saved {lead.name} to Supabase")

    output = LeadOutput(
        name=lead.name,
        score=final_state["score"],
        score_reason=final_state["score_reason"],
        decision=final_state["decision"],
        reason=final_state["reason"],
        email_type=email_type,
        email_confidence=email_confidence,
        email_reason=email_reason,
        followup_email=followup_email,
        email_sent=email_sent
    )

    send_outbound_webhook(output.model_dump())

    return output


@app.get("/")
def read_root():
    return {"message": "AI Lead Qualifier API is running"}


@app.post("/qualify", response_model=LeadOutput)
@limiter.limit("10/hour")
def qualify_lead(request: Request, lead: LeadInput, api_key: str = Depends(verify_api_key)):
    return process_lead(lead)


@app.post("/submit-lead", response_model=LeadOutput)
@limiter.limit("10/hour")
def submit_lead(request: Request, lead: LeadInput):
    client_ip = get_client_ip(request)
    if not check_and_increment_ip_usage(client_ip):
        raise HTTPException(
            status_code=429,
            detail="You've reached the maximum number of free submissions from this location."
        )
    return process_lead(lead)


@app.get("/leads")
def get_leads():
    response = supabase.table("leads").select("*").order("created_at", desc=True).execute()
    return response.data


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
        const response = await fetch("/submit-lead", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (response.status === 429) {
            const errorData = await response.json();
            resultDiv.style.display = "block";
            resultDiv.className = "error";
            resultDiv.textContent = errorData.detail || "Too many submissions. Please try again later.";
            submitBtn.disabled = false;
            submitBtn.textContent = "Submit";
            return;
        }

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
        html += "<strong>Email type:</strong> " + data.email_type + " (" + data.email_confidence + "% confidence) - " + data.email_reason + "<br>";
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


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Lead Dashboard</title>
<style>
    body {
        font-family: Arial, sans-serif;
        max-width: 1100px;
        margin: 40px auto;
        padding: 20px;
        background: #f5f5f5;
    }
    h1 {
        font-size: 22px;
        margin-bottom: 20px;
    }
    .controls {
        margin-bottom: 16px;
        display: flex;
        gap: 12px;
        align-items: center;
    }
    select {
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    th, td {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
        font-size: 13px;
    }
    th {
        background: #222;
        color: white;
        cursor: pointer;
        user-select: none;
    }
    th:hover {
        background: #444;
    }
    tr:hover {
        background: #fafafa;
    }
    .badge {
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge.qualified {
        background: #e6f4ea;
        color: #34a853;
    }
    .badge.nurture {
        background: #fef7e0;
        color: #b06000;
    }
    .badge.error {
        background: #fce8e6;
        color: #ea4335;
    }
    .badge.business {
        background: #e8f0fe;
        color: #1a73e8;
    }
    .badge.personal {
        background: #f1f3f4;
        color: #5f6368;
    }
    .badge.disposable {
        background: #fce8e6;
        color: #ea4335;
    }
    #loading {
        padding: 20px;
        color: #666;
    }
</style>
</head>
<body>

<h1>Lead Dashboard</h1>

<div class="controls">
    <label for="filterSelect">Filter:</label>
    <select id="filterSelect">
        <option value="all">All</option>
        <option value="Qualified">Qualified</option>
        <option value="Nurture">Nurture</option>
    </select>
</div>

<div id="loading">Loading leads...</div>
<table id="leadsTable" style="display: none;">
    <thead>
        <tr>
            <th data-key="name">Name</th>
            <th data-key="email">Email</th>
            <th data-key="score">Score &#8645;</th>
            <th data-key="email_type">Email Type</th>
            <th data-key="source">Source</th>
            <th data-key="ai_decision">Decision</th>
            <th data-key="ai_reason">Reason</th>
            <th data-key="email_sent">Follow-up Sent</th>
            <th data-key="created_at">Date &#8645;</th>
        </tr>
    </thead>
    <tbody id="leadsBody"></tbody>
</table>

<script>
let allLeads = [];
let sortKey = "created_at";
let sortAsc = false;

async function loadLeads() {
    const response = await fetch("/leads");
    allLeads = await response.json();
    document.getElementById("loading").style.display = "none";
    document.getElementById("leadsTable").style.display = "table";
    renderTable();
}

function renderTable() {
    const filterValue = document.getElementById("filterSelect").value;
    let leads = allLeads;

    if (filterValue !== "all") {
        leads = leads.filter(l => l.ai_decision === filterValue);
    }

    leads = leads.slice().sort((a, b) => {
        let valA = a[sortKey];
        let valB = b[sortKey];
        if (valA === null || valA === undefined) valA = "";
        if (valB === null || valB === undefined) valB = "";
        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
    });

    const tbody = document.getElementById("leadsBody");
    tbody.innerHTML = "";

    leads.forEach(lead => {
        const row = document.createElement("tr");

        const decisionClass = lead.ai_decision === "Qualified" ? "qualified" :
                               lead.ai_decision === "Nurture" ? "nurture" : "error";

        const emailTypeClass = lead.email_type === "business" ? "business" :
                                lead.email_type === "disposable" ? "disposable" : "personal";

        const dateStr = lead.created_at ? new Date(lead.created_at).toLocaleString() : "";

        row.innerHTML = `
            <td>${lead.name || ""}</td>
            <td>${lead.email || ""}</td>
            <td>${lead.score ?? ""}</td>
            <td><span class="badge ${emailTypeClass}">${lead.email_type || ""}</span></td>
            <td>${lead.source || ""}</td>
            <td><span class="badge ${decisionClass}">${lead.ai_decision || ""}</span></td>
            <td>${lead.ai_reason || ""}</td>
            <td>${lead.email_sent ? "Yes" : "No"}</td>
            <td>${dateStr}</td>
        `;
        tbody.appendChild(row);
    });
}

document.getElementById("filterSelect").addEventListener("change", renderTable);

document.querySelectorAll("th[data-key]").forEach(th => {
    th.addEventListener("click", () => {
        const key = th.getAttribute("data-key");
        if (sortKey === key) {
            sortAsc = !sortAsc;
        } else {
            sortKey = key;
            sortAsc = true;
        }
        renderTable();
    });
});

loadLeads();
</script>

</body>
</html>
"""