from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import random
from typing import Dict

app = FastAPI(title="SceneFlow Admin API - Fast Switch")

# Environment variables
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]

if not ADMIN_TOKEN:
    raise ValueError("ADMIN_TOKEN environment variable is required!")
if not GEMINI_KEYS:
    raise ValueError("GEMINI_API_KEYS environment variable is required! (comma separated keys)")

# Track cooldown per key (কমানো হয়েছে যাতে ফাস্ট সুইচ হয়)
key_cooldown: Dict[str, datetime] = {key: datetime.utcnow() - timedelta(days=1) for key in GEMINI_KEYS}
COOLDOWN_BASE = 5  # ১ মিনিট (আপনার চাহিদা অনুযায়ী কমানো, চাইলে 30 করুন)

current_key_index = 0

def get_next_key() -> str:
    global current_key_index
    n = len(GEMINI_KEYS)
    attempts = 0

    while attempts < n * 3:  # আরও aggressive খোঁজা (৩ গুণ চেষ্টা)
        key = GEMINI_KEYS[current_key_index % n]
        if datetime.utcnow() >= key_cooldown[key]:
            current_key_index = (current_key_index + 1) % n
            return key
        current_key_index = (current_key_index + 1) % n
        attempts += 1

    # সব exhausted → ছোট wait + auto-retry hint
    raise HTTPException(
        429,
        "All keys temporarily exhausted. Automatically retrying soon... (wait 1-2 min max)"
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

    for attempt in range(12):  # ১২ বার চেষ্টা (ফাস্ট সুইচের জন্য বাড়ানো)
        api_key = get_next_key()

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(request.model)

            gen_config = {"temperature": request.temperature}
            if request.max_tokens:
                gen_config["max_output_tokens"] = request.max_tokens

            response = model.generate_content(request.prompt, generation_config=gen_config)

            # সফল হলে cooldown reset
            key_cooldown[api_key] = datetime.utcnow() - timedelta(days=1)

            return {
                "status": "success",
                "result": response.text,
                "model_used": request.model,
                "key_hint": api_key[:8] + "...",
                "attempt": attempt + 1
            }

        except Exception as e:
            str_e = str(e).lower()
            if "429" in str_e or "quota" in str_e or "exhausted" in str_e or "rate limit" in str_e:
                cooldown_time = COOLDOWN_BASE + random.randint(0, 30)  # ৬০-৯০ সেকেন্ড
                key_cooldown[api_key] = datetime.utcnow() + timedelta(seconds=cooldown_time)
                print(f"Key {api_key[:8]}... exhausted → cooldown {cooldown_time} sec, auto-switching...")
                # চেষ্টা চালিয়ে যান (অটো সুইচ)
            else:
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    raise HTTPException(
        429,
        "Quota exhausted on all keys after max retries. Wait 1-2 min and try again (auto-retrying enabled)."
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "keys_loaded": len(GEMINI_KEYS), "cooldown_active": any(datetime.utcnow() < t for t in key_cooldown.values())}
