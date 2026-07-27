from langgraph.graph import StateGraph, END
from typing import TypedDict


class LeadState(TypedDict):
    name: str
    score: int
    decision: str


def check_score(state: LeadState) -> LeadState:
    print(f"Checking score for {state['name']}...")
    return state


def route_decision(state: LeadState) -> str:
    if state["score"] >= 70:
        return "qualify"
    else:
        return "nurture"


def qualify_node(state: LeadState) -> LeadState:
    state["decision"] = "Qualified"
    print(f"{state['name']} -> Qualified")
    return state


def nurture_node(state: LeadState) -> LeadState:
    state["decision"] = "Nurture"
    print(f"{state['name']} -> Nurture")
    return state


graph = StateGraph(LeadState)

graph.add_node("check_score", check_score)
graph.add_node("qualify", qualify_node)
graph.add_node("nurture", nurture_node)

graph.set_entry_point("check_score")

graph.add_conditional_edges(
    "check_score",
    route_decision,
    {
        "qualify": "qualify",
        "nurture": "nurture"
    }
)

graph.add_edge("qualify", END)
graph.add_edge("nurture", END)

app = graph.compile()

result = app.invoke({"name": "Ion Popescu", "score": 85, "decision": ""})
print(result)

result2 = app.invoke({"name": "Maria Ionescu", "score": 45, "decision": ""})
print(result2)