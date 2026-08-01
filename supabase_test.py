import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

result = supabase.table("leads").insert({
    "name": "Test Lead",
    "email": "test@example.com",
    "score": 99,
    "source": "Manual Test",
    "ai_decision": "Qualified",
    "ai_reason": "This is just a connection test"
}).execute()

print(result)