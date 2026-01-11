# SceneFlow Admin API (Render-ready)

এই সার্ভারটা আপনার নিজের Gemini API key ব্যবহার করে একটি **Admin API** দেয়।
আপনার সফটওয়্যারে ইউজার চাইলে:
- নিজের Gemini key দিয়ে direct চলতে পারবে, অথবা
- আপনার এই Admin API ব্যবহার করতে পারবে (Gemini key ইউজারকে দিতে হবে না)

## Endpoints
- `GET /health` → সার্ভার চলছে কিনা
- `GET /v1/models` → allowlist মডেল লিস্ট (Authorization লাগে)
- `POST /v1/generate` → টেক্সট generate (Authorization লাগে)

Authorization header:
`Authorization: Bearer <ADMIN_TOKEN>`

## Required Environment Variables (Render dashboard → Environment)
- `ADMIN_TOKEN` **অথবা** `ADMIN_TOKENS` (comma-separated)
- `GEMINI_API_KEY`

Optional:
- `ALLOWED_MODELS` (comma-separated model ids)
- `RATE_LIMIT_PER_MIN` (default 30)
- `MAX_PROMPT_CHARS` (default 12000)
- `REQUEST_TIMEOUT_SEC` (default 60)

## Local run (optional)
```bash
pip install -r requirements.txt
export ADMIN_TOKEN="your-admin-token"
export GEMINI_API_KEY="your-gemini-key"
uvicorn main:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -H "Authorization: Bearer your-admin-token" http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/generate \
  -H "Authorization: Bearer your-admin-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-flash-latest","prompt":"Hello","temperature":0.7}'
```

## Deploy on Render (Free)
1) GitHub repo বানান এবং এই ফাইলগুলো push করুন
2) Render → **New +** → **Web Service** → GitHub repo select
3) Build Command: `pip install -r requirements.txt`
4) Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5) Environment tab-এ `ADMIN_TOKEN`, `GEMINI_API_KEY` সেট করুন
6) Deploy

> Note: Free tier এ service idle হলে spin-down হতে পারে, প্রথম request এ কিছু delay আসতে পারে।
