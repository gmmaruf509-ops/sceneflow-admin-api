from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import google.generativeai as genai
import os
import random
from datetime import datetime, timedelta
from typing import Dict

app = FastAPI()

# Environment variables
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]

if not ADMIN_TOKEN or not GEMINI_KEYS:
    raise ValueError("Missing ADMIN_TOKEN or GEMINI_API_KEYS")

# Key status tracking
key_status: Dict[str, datetime] = {key: datetime.min for key in GEMINI_KEYS}
COOLDOWN_SECONDS = 300 + random.randint(0, 120)  # 5-7 min

class GenerateRequest(BaseModel):
    model: str = "gemini-flash-latest"
    prompt: str
    temperature: float = 0.7

def get_available_key():
    for _ in range(len(GEMINI_KEYS) * 2):  # prevent infinite loop
        key = random.choice(GEMINI_KEYS)  # or round-robin
        if datetime.now() >= key_status[key]:
            return key
    raise HTTPException(429, "All keys temporarily exhausted. Wait 5-10 min.")

@app.post("/v1/generate")
async def generate(request: GenerateRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Invalid token")

    for attempt in range(5):
        api_key = get_available_key()

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(request.model)
            response = model.generate_content(
                request.prompt,
                generation_config={"temperature": request.temperature}
            )
            return {"result": response.text, "used_key": api_key[:8]+"..."}

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "exhausted" in err:
                key_status[api_key] = datetime.now() + timedelta(seconds=COOLDOWN_SECONDS)
                if attempt == 4:
                    raise HTTPException(429, "Quota exhausted on all available keys")
            else:
                raise HTTPException(500, str(e))

    raise HTTPException(500, "Failed after retries")
