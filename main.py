from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import random
from typing import Dict

app = FastAPI(title="SceneFlow Admin API - Ultra Fast Switch")

# Environment variables
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]

if not ADMIN_TOKEN:
    raise ValueError("ADMIN_TOKEN environment variable is required!")
if not GEMINI_KEYS:
    raise ValueError("GEMINI_API_KEYS environment variable is required! (comma separated keys)")

# Cooldown খুব কম (১০ সেকেন্ড) — আপনার চাহিদা অনুযায়ী
key_cooldown: Dict[str, datetime] = {key: datetime.utcnow() - timedelta(days=1) for key in GEMINI_KEYS}
COOLDOWN_BASE = 10  # ১০ সেকেন্ড — চাইলে 5 বা 1 করুন (রিস্ক বাড়বে)

current_key_index = 0

def get_next_key() -> str:
    global current_key_index
    n = len(GEMINI_KEYS)
    attempts = 0

    while attempts < n * 5:  # আরও বেশি চেষ্টা — ফাস্ট সুইচের জন্য
        key = GEMINI_KEYS[current_key_index % n]
        if datetime.utcnow() >= key_cooldown[key]:
            current_key_index = (current_key_index + 1) % n
            return key
        current_key_index = (current_key_index + 1) % n
        attempts += 1

    # সব exhausted → খুব ছোট wait + auto-retry hint
    raise HTTPException(
        429,
        "All keys temporarily exhausted. Automatically retrying in seconds..."
    )

class GenerateRequest(BaseModel):
    model: str = "gemini-flash-latest"
    prompt: str
    temperature: float = 0.7
    max_tokens: int | None = None

@app.post("/v1/generate")
async def generate(
    request: GenerateRequest = Body(...),
    authorization: str = Header(None, alias="Authorization")
):
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization token")

    max_attempts = 20  # অনেক বেশি চেষ্টা — যাতে পরের key পায়
    for attempt in range(max_attempts):
        api_key = get_next_key()

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(request.model)

            gen_config = {"temperature": request.temperature}
            if request.max_tokens:
                gen_config["max_output_tokens"] = request.max_tokens

            response = model.generate_content(request.prompt, generation_config=gen_config)

            # সফল হলে reset
            key_cooldown[api_key] = datetime.utcnow() - timedelta(days=1)

            return {
                "status": "success",
                "result": response.text,
                "model_used": request.model,
                "key_hint": api_key[:8] + "...",
                "attempt": attempt + 1,
                "note": "Switched automatically if needed"
            }

        except Exception as e:
            str_e = str(e).lower()
            if "429" in str_e or "quota" in str_e or "exhausted" in str_e or "rate limit" in str_e:
                cooldown_time = COOLDOWN_BASE + random.randint(0, 10)  # ১০-২০ সেকেন্ড
                key_cooldown[api_key] = datetime.utcnow() + timedelta(seconds=cooldown_time)
                print(f"Key {api_key[:8]}... exhausted → cooldown {cooldown_time}s, switching immediately...")
                # অটো চালিয়ে যান
            else:
                raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    raise HTTPException(
        429,
        "Quota exhausted on all keys after max fast retries. Wait 30-60 seconds and try again (auto-retrying)."
    )

@app.get("/health")
async def health():
    active_cooldowns = sum(1 for t in key_cooldown.values() if datetime.utcnow() < t)
    return {
        "status": "healthy",
        "keys_loaded": len(GEMINI_KEYS),
        "keys_in_cooldown": active_cooldowns
    }
