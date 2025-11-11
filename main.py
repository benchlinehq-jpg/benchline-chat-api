import os
from typing import List, Literal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CORS (who can call your API) ---
ALLOWED = os.getenv("ALLOWED_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in ALLOWED.split(",")] if ALLOWED else ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
Role = Literal["user", "assistant"]
class Msg(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Msg]

# --- Health check ---
@app.get("/healthz")
def healthz():
    return {"ok": True}

# --- Rule-based fallback (works even without OpenAI key) ---
def rule_based_answer(user_text: str) -> str:
    t = user_text.strip().lower()
    if t in {"hi", "hello", "hey"}: return "Hey! I’m Benchline Bot. Try asking: what's 2+2?"
    if "2+2" in t or "2 + 2" in t:  return "2 + 2 = 4."
    if "name" in t:                 return "I’m Benchline Bot."
    if "hours" in t:                return "I’m awake 24/7 🙂"
    return f"You said: “{user_text}”. (Test reply—backend working!)"

# --- OpenAI client (optional) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# --- Chat endpoint ---
@app.post("/api/chat")
def chat(req: ChatRequest):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not client:
        return JSONResponse({"reply": rule_based_answer(last_user)})

    # Use OpenAI when key is present
    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are Benchline Bot. Be concise, friendly, and helpful."}]
                     + [m.model_dump() for m in req.messages],
            temperature=0.2,
        )
        text = out.choices[0].message.content
        return JSONResponse({"reply": text})
    except Exception:
        # graceful fallback so chat still works if API call fails
        return JSONResponse({"reply": rule_based_answer(last_user) + "\n\n[Note: model fallback]"})
