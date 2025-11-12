# ========== Benchline Chat API (FastAPI) ==========
import os, sys, time, json, csv, datetime
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional: OpenAI client (only used if OPENAI_API_KEY is set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# ---------------- App + CORS ----------------
ALLOWED = os.getenv("ALLOWED_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in ALLOWED.split(",") if o.strip()]
if not ALLOW_ORIGINS:
    ALLOW_ORIGINS = ["https://benchlinehq-jpg.github.io"]  # safe default

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------- Tiny per-IP rate limit ----------------
BUCKET = {}  # ip -> [timestamps]

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    try:
        ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host
    except Exception:
        ip = "unknown"
    now = time.time()
    window = 60.0
    limit = 20

    q = BUCKET.get(ip, [])
    q = [t for t in q if now - t < window]
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    q.append(now)
    BUCKET[ip] = q
    return await call_next(request)

# ---------------- Logging helper ----------------
def log(event: str, **kw):
    print(json.dumps({"event": event, **kw}), file=sys.stdout, flush=True)

# ---------------- Models ----------------
Role = Literal["user", "assistant"]

class Msg(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Msg]

class Lead(BaseModel):
    name: str
    email: str           # keep simple (no extra dependency)
    message: str | None = ""
    source: str | None = "chat-widget"

# ---------------- Health ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------------- Rule-based fallback ----------------
def rule_based_answer(user_text: str) -> str:
    t = (user_text or "").strip().lower()
    if t in {"hi", "hello", "hey"}: return "Hey! I’m Benchline Bot. Try asking: what's 2+2?"
    if "2+2" in t or "2 + 2" in t:  return "2 + 2 = 4."
    if "name" in t:                 return "I’m Benchline Bot."
    if "hours" in t:                return "I’m awake 24/7 🙂"
    return f"You said: “{user_text}”. (Test reply—backend working!)"

# ---------------- FAQ fast-answer layer ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(HERE, "faq.json")
FAQ_MIN_SCORE = int(os.getenv("FAQ_MIN_SCORE", "1"))

def load_faq():
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                item["q"] = [str(x).lower().strip() for x in item.get("q", [])]
                item["a"] = str(item.get("a", "")).strip()
            return data
    except Exception as e:
        log("faq_load_error", err=str(e))
        return []

FAQ = load_faq()

def faq_answer(user_text: str) -> str | None:
    if not FAQ:
        return None
    t = (user_text or "").lower()
    best = None
    best_score = 0
    for item in FAQ:
        score = sum(1 for kw in item["q"] if kw and kw in t)
        if score > best_score:
            best_score, best = score, item
    if best and best_score >= FAQ_MIN_SCORE and best.get("a"):
        return best["a"]
    return None

# ---------------- Chat endpoint ----------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()
    log("chat_in", user=last_user)

    # 1) FAQ instant
    fa = faq_answer(last_user)
    if fa:
        log("chat_out", reply=fa, mode="faq")
        return JSONResponse({"reply": fa})

    # 2) OpenAI or fallback
    if not client:
        reply = rule_based_answer(last_user)
        log("chat_out", reply=reply, mode="fallback")
        return JSONResponse({"reply": reply})

    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are Benchline Bot. Be concise, friendly, and helpful."}]
                     + [m.model_dump() for m in req.messages],
            temperature=0.2,
        )
        text = out.choices[0].message.content
        log("chat_out", reply=text, mode="openai")
        return JSONResponse({"reply": text})
    except Exception as e:
        log("chat_error", err=str(e))
        reply = rule_based_answer(last_user) + "\n\n[Note: model fallback]"
        log("chat_out", reply=reply, mode="fallback_error")
        return JSONResponse({"reply": reply})

# ---------------- Lead capture ----------------
# CSV backup path (leave empty to disable file writes)
LEADS_PATH = os.getenv("LEADS_CSV", "/tmp/leads.csv")

# Optional Zapier webhook (set in Render env as LEAD_WEBHOOK)
import requests  # ensure requests==2.32.3 in requirements.txt

@app.post("/api/lead")
def capture_lead(lead: Lead):
    # Normalize
    name   = (lead.name or "").strip()
    email  = (lead.email or "").strip().lower()
    msg    = (lead.message or "").strip()
    source = (lead.source or "chat-widget").strip()
    ts     = datetime.datetime.utcnow().isoformat() + "Z"

    # Spam guard
    bad_emails = {"test@test.com", "example@example.com"}
    if len(name) < 2 or "http://" in name.lower() or "https://" in name.lower():
        return JSONResponse(content={"ok": False, "error": "invalid_name"}, status_code=400)
    if "@" not in email or email in bad_emails or email.endswith("@example.com"):
        return JSONResponse(content={"ok": False, "error": "invalid_email"}, status_code=400)
    if msg and len(msg) > 1000:
        msg = msg[:1000]

    # CSV backup (if enabled)
    if LEADS_PATH:
        try:
            file_exists = os.path.isfile(LEADS_PATH)
            with open(LEADS_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(["ts_utc", "name", "email", "message", "source"])
                w.writerow([ts, name, email, msg, source])
        except Exception as e:
            log("lead_write_error", err=str(e))
            log("lead_fallback", ts=ts, name=name, email=email, message=msg, source=source)

    # Zapier webhook
    try:
        hook = os.getenv("LEAD_WEBHOOK", "").strip()
        if hook:
            payload = {"ts": ts, "name": name, "email": email, "message": msg, "source": source}
            requests.post(hook, json=payload, timeout=5)
    except Exception as e:
        log("lead_webhook_error", err=str(e))

    log("lead_in", name=name, email=email, source=source)
    return {"ok": True}

# --- Download leads CSV (quick admin endpoint) ---
@app.get("/api/leads.csv")
def get_leads_csv():
    path = LEADS_PATH
    if not path or not os.path.isfile(path):
        return PlainTextResponse("ts_utc,name,email,message,source\n", media_type="text/csv")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return PlainTextResponse(data, media_type="text/csv")
