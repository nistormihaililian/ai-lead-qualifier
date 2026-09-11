import pytest
from api import (
    parse_score,
    parse_response,
    parse_email_analysis,
    route_after_score,
    route_after_call,
    MAX_RETRIES
)


def make_state(**overrides):
    base = {
        "name": "Test Lead",
        "email": "test@example.com",
        "source": "Test Source",
        "message": "Test message",
        "email_type": "personal",
        "email_confidence": 50,
        "email_reason": "",
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
    base.update(overrides)
    return base


def test_parse_score_valid_response():
    state = make_state(raw_score_response="SCORE: 85 | REASON: Strong intent and clear budget")
    result = parse_score(state)
    assert result["score"] == 85
    assert result["score_reason"] == "Strong intent and clear budget"


def test_parse_score_malformed_response():
    state = make_state(raw_score_response="this is not a valid format")
    result = parse_score(state)
    assert result["score"] == 0
    assert "Could not parse" in result["score_reason"]


def test_parse_response_qualified():
    state = make_state(raw_response="DECISION: Qualified | REASON: High score and urgency")
    result = parse_response(state)
    assert result["decision"] == "Qualified"
    assert result["reason"] == "High score and urgency"


def test_parse_response_nurture():
    state = make_state(raw_response="DECISION: Nurture | REASON: Low intent detected")
    result = parse_response(state)
    assert result["decision"] == "Nurture"


def test_parse_response_malformed():
    state = make_state(raw_response="garbage output")
    result = parse_response(state)
    assert result["decision"] == "Error"


def test_parse_email_analysis_business():
    raw = "TYPE: business | CONFIDENCE: 90 | REASON: Custom company domain"
    email_type, confidence, reason = parse_email_analysis(raw)
    assert email_type == "business"
    assert confidence == 90
    assert reason == "Custom company domain"


def test_parse_email_analysis_disposable():
    raw = "TYPE: disposable | CONFIDENCE: 95 | REASON: Known temporary email provider"
    email_type, confidence, reason = parse_email_analysis(raw)
    assert email_type == "disposable"
    assert confidence == 95


def test_parse_email_analysis_malformed():
    raw = "not a valid response"
    email_type, confidence, reason = parse_email_analysis(raw)
    assert email_type == "unknown"
    assert confidence == 0


def test_route_after_score_success():
    state = make_state(score_failed=False)
    assert route_after_score(state) == "parse_score"


def test_route_after_score_retry():
    state = make_state(score_failed=True, score_attempt=1)
    assert route_after_score(state) == "retry_score"


def test_route_after_score_give_up():
    state = make_state(score_failed=True, score_attempt=MAX_RETRIES)
    assert route_after_score(state) == "give_up_score"


def test_route_after_call_success():
    state = make_state(failed=False)
    assert route_after_call(state) == "parse"


def test_route_after_call_retry():
    state = make_state(failed=True, attempt=1)
    assert route_after_call(state) == "retry"


def test_route_after_call_give_up():
    state = make_state(failed=True, attempt=MAX_RETRIES)
    assert route_after_call(state) == "give_up"