from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import random
from typing import Dict, Any

app = FastAPI(title="SceneFlow Admin API")

# Environment variables
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]

if not ADMIN_TOKEN:
    raise ValueError("ADMIN_TOKEN environment variable is required!")
if not GEMINI_KEYS:
    raise ValueError("GEMINI_API_KEYS environment variable is required! (comma separated keys)")

# Track when each key can be tried again
key_cooldown: Dict[str, datetime] = {key: datetime.utcnow() - timedelta(days=1) for key in GEMINI_KEYS}
COOLDOWN_BASE = 300  # 5 minutes

current_key_index = 0

def get_next_key() -> str:
    global current_key_index
    attempts = 0
    n = len(GEMINI_KEYS)

    while attempts < n:
        key = GEMINI_KEYS[current_key_index % n]
        if datetime.utcnow() >= key_cooldown[key]:
            current_key_index = (current_key_index + 1) % n
            return key
        current_key_index = (current_key_index + 1) % n
        attempts += 1

    # All keys cooled down → estimate wait time
    waits = [(key_cooldown[k] - datetime.utcnow()).total_seconds() for k in GEMINI_KEYS]
    max_wait = max(waits) if waits else 60
    raise HTTPException(429, f"All {n} keys exhausted. Retry after ~{int(max_wait / 60)} minutes")

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

    for attempt in range(len(GEMINI_KEYS)):
        api_key = get_next_key()

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(request.model)

            gen_config = {"temperature": request.temperature}
            if request.max_tokens:
                gen_config["max_output_tokens"] = request.max_tokens

            response = model.generate_content(request.prompt, generation_config=gen_config)

            # Success → reset cooldown for this key
            key_cooldown[api_key] = datetime.utcnow() - timedelta(days=1)

            return {
                "status": "success",
                "result": response.text,
                "model_used": request.model,
                "key_hint": api_key[:8] + "..."
            }

        except Exception as e:
            str_e = str(e).lower()
            if "429" in str_e or "quota" in str_e or "exhausted" in str_e or "rate limit" in str_e:
                # Apply cooldown
                cooldown_time = COOLDOWN_BASE + random.randint(0, 120)
                key_cooldown[api_key] = datetime.utcnow() + timedelta(seconds=cooldown_time)
                print(f"Key {api_key[:8]}... hit limit → cooldown {cooldown_time//60} min")
            else:
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    raise HTTPException(status_code=429, detail="All available keys exhausted after retries")

@app.get("/health")
async def health():
    return {"status": "healthy", "keys_loaded": len(GEMINI_KEYS)}
