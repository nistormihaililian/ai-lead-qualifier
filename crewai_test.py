import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()

copywriter = Agent(
    role="Sales Follow-up Copywriter",
    goal="Write short, personalized follow-up messages for qualified sales leads",
    backstory="You are an experienced sales copywriter who writes concise, "
               "friendly follow-up emails that get replies, without sounding pushy.",
    llm="anthropic/claude-haiku-4-5-20251001",
    verbose=True
)

write_followup = Task(
    description="Write a short follow-up email (max 60 words) for this lead: "
                 "Name: Ion Popescu, Source: Facebook Ads, "
                 "Qualification reason: High score indicates strong purchase intent. "
                 "Keep it warm, direct, and mention the source naturally.",
    expected_output="A short follow-up email, max 60 words, with a subject line.",
    agent=copywriter
)

crew = Crew(
    agents=[copywriter],
    tasks=[write_followup],
    verbose=True
)

result = crew.kickoff()
print("\n--- FINAL RESULT ---")
print(result)