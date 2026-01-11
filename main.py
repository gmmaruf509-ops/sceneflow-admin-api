from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# -----------------------------
# Environment / Config
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


# Required
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# You can set one token (ADMIN_TOKEN) or a comma-separated list (ADMIN_TOKENS)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_TOKENS = os.getenv("ADMIN_TOKENS", "")

# Optional
ALLOWED_MODELS = os.getenv(
    "ALLOWED_MODELS",
    # Default allowlist (edit as you want)
    "gemini-flash-latest,gemini-flash-lite-latest,gemini-2.0-flash-001,gemini-2.0-flash,gemini-2.0-flash-exp,"
    "gemini-2.0-flash-lite-001,gemini-2.0-flash-lite,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.5-pro,gemini-pro-latest",
)

# Rate limit (simple in-memory; resets on restart)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))  # per token or per IP
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "12000"))
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "60"))

# Gemini REST endpoint base
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")


def get_allowed_models() -> List[str]:
    models = [m.strip() for m in ALLOWED_MODELS.split(",") if m.strip()]
    # Keep unique order
    seen = set()
    out = []
    for m in models:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def get_admin_tokens() -> List[str]:
    tokens: List[str] = []
    if ADMIN_TOKEN.strip():
        tokens.append(ADMIN_TOKEN.strip())
    if ADMIN_TOKENS.strip():
        tokens.extend([t.strip() for t in ADMIN_TOKENS.split(",") if t.strip()])
    # unique
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# -----------------------------
# Simple in-memory rate limiter
# -----------------------------
class RateLimiter:
    """
    Simple fixed-window limiter: N requests per 60 seconds per key.
    Not perfect, but enough to protect your billing from abuse.
    """
    def __init__(self, limit_per_minute: int):
        self.limit = max(1, int(limit_per_minute))
        self._buckets: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)

    def check(self, key: str) -> None:
        now = time.time()
        count, start = self._buckets.get(key, (0, now))
        # New window?
        if now - start >= 60:
            count, start = 0, now
        count += 1
        self._buckets[key] = (count, start)
        if count > self.limit:
            retry_after = max(1, int(60 - (now - start)))
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limited", "retry_after_sec": retry_after, "limit_per_minute": self.limit},
                headers={"Retry-After": str(retry_after)},
            )


limiter = RateLimiter(RATE_LIMIT_PER_MIN)

app = FastAPI(title="SceneFlow Admin API", version="1.0.0")


# -----------------------------
# Schemas
# -----------------------------
class GenerateRequest(BaseModel):
    model: str = Field(..., description="Gemini model id (must be in allowlist).")
    prompt: str = Field(..., description="User prompt text.")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_output_tokens: int = Field(2048, ge=1, le=8192)


class GenerateResponse(BaseModel):
    ok: bool
    model: str
    text: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Helpers
# -----------------------------
def _get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _auth_or_401(authorization: Optional[str]) -> str:
    tokens = get_admin_tokens()
    if not tokens:
        # Misconfigured server
        raise HTTPException(status_code=500, detail="Server not configured: set ADMIN_TOKEN or ADMIN_TOKENS.")
    token = _get_bearer_token(authorization)
    if not token or token not in tokens:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


def _client_key(request: Request, token: str) -> str:
    # Rate limit key: token + client IP
    ip = request.client.host if request.client else "unknown"
    return f"{token}:{ip}"


async def _call_gemini_generate(req: GenerateRequest) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server not configured: set GEMINI_API_KEY.")

    url = f"{GEMINI_BASE_URL}/models/{req.model}:generateContent"
    params = {"key": GEMINI_API_KEY}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
        "generationConfig": {
            "temperature": req.temperature,
            "topP": req.top_p,
            "maxOutputTokens": req.max_output_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC) as client:
        r = await client.post(url, params=params, json=payload)
        if r.status_code >= 400:
            try:
                data = r.json()
            except Exception:
                data = {"error": r.text}
            raise HTTPException(status_code=r.status_code, detail=data)
        return r.json()


def _extract_text(gemini_json: Dict[str, Any]) -> str:
    try:
        candidates = gemini_json.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: List[str] = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    _auth_or_401(authorization)
    return {"ok": True, "models": get_allowed_models()}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request, authorization: Optional[str] = Header(default=None)):
    token = _auth_or_401(authorization)

    # Rate limiting
    limiter.check(_client_key(request, token))

    # Validation
    if body.model not in get_allowed_models():
        raise HTTPException(status_code=400, detail={"error": "model_not_allowed", "allowed_models": get_allowed_models()})

    prompt = body.prompt or ""
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail={"error": "prompt_too_large", "max_chars": MAX_PROMPT_CHARS})

    gemini_json = await _call_gemini_generate(body)
    text = _extract_text(gemini_json)

    return GenerateResponse(ok=True, model=body.model, text=text, raw=gemini_json)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    headers = getattr(exc, "headers", None) or {}
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail}, headers=headers)
