import asyncio
import base64
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("negotiation_coach")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GRADIUM_API_KEY = os.environ.get("GRADIUM_API_KEY", "")
GRADIUM_BASE_URL = "https://api.gradium.ai/api"

# Module-level singletons populated in lifespan().
# See PERFORMANCE.md → "Shared clients" for why these are not per-request.
_http_client: Optional[httpx.AsyncClient] = None
_anthropic_client: Optional[anthropic.AsyncAnthropic] = None
_index_html: str = ""
_index_etag: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _anthropic_client, _index_html, _index_etag

    _http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    _anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    with open("index.html", "rb") as f:
        raw = f.read()
    _index_html = raw.decode()
    # Weak ETag — content-based so any edit to index.html invalidates the browser cache.
    _index_etag = 'W/"' + hashlib.sha256(raw).hexdigest()[:16] + '"'

    logger.info(
        "startup complete (index.html=%d bytes, etag=%s)", len(raw), _index_etag
    )

    try:
        yield
    finally:
        await _http_client.aclose()


app = FastAPI(lifespan=lifespan)

# GZip after CORS so preflight stays uncompressed and small.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

VOICES = [
    {
        "id": "YTpq7expH9539ERJ",
        "name": "Emma",
        "role": "HR Manager",
        "accent": "American",
        "style": "Warm & Approachable",
        "color": "#6366f1",
        "image": "/static/images/emma.png",
    },
    {
        "id": "LFZvm12tW_z0xfGo",
        "name": "Kent",
        "role": "Senior Recruiter",
        "accent": "American",
        "style": "Professional & Direct",
        "color": "#0ea5e9",
        "image": "/static/images/kent.png",
    },
    {
        "id": "ubuXFxVQwVYnZQhy",
        "name": "Eva",
        "role": "Director of HR",
        "accent": "British",
        "style": "Formal & Precise",
        "color": "#ec4899",
        "image": "/static/images/eva.png",
    },
    {
        "id": "m86j6D7UZpGzHsNu",
        "name": "Jack",
        "role": "VP of Talent",
        "accent": "British",
        "style": "Firm & Experienced",
        "color": "#f59e0b",
        "image": "/static/images/jack.png",
    },
]

DIFFICULTIES = {
    "easy": {
        "label": "Easy",
        "emoji": "😊",
        "color": "#22c55e",
        "instruction": "Be friendly and accommodating. Start 20% below the maximum budget. Move up quickly with 1-2 good arguments from the candidate. Be encouraging and supportive.",
    },
    "medium": {
        "label": "Medium",
        "emoji": "🤔",
        "color": "#f59e0b",
        "instruction": "Be professional but firm. Start 30% below the maximum budget. Require 2-3 solid arguments before making concessions of 5-10% each.",
    },
    "hard": {
        "label": "Hard",
        "emoji": "😤",
        "color": "#ef4444",
        "instruction": "Be tough and skeptical. Start 40% below the maximum budget. Only concede 5% at a time with very compelling market data. Use budget constraints as barriers.",
    },
    "impossible": {
        "label": "Impossible",
        "emoji": "😈",
        "color": "#7c3aed",
        "instruction": "Be relentless. Start 50% below market rate. Maximum 3% total movement. Use: 'other candidates asked for less', 'budget is frozen', 'this is non-negotiable'. Never show sympathy.",
    },
}


def build_system_prompt(job_description: str, difficulty: str, voice: dict) -> str:
    d = DIFFICULTIES[difficulty]
    return f"""You are {voice["name"]}, {voice["role"]} at a well-funded tech company. Style: {voice["style"]}.

JOB POSITION:
{job_description}

APPROACH: {d["instruction"]}

RULES:
- Under 150 words, conversational, natural for speech
- Always use specific numbers (salary, equity %, bonus $)
- Never go below your last stated number
- Never reveal you are an AI
- React to the candidate's actual arguments

Make your OPENING OFFER now — include specific salary, equity, and bonus numbers."""


async def tts(text: str, voice_id: str) -> Optional[str]:
    if not GRADIUM_API_KEY:
        return None
    try:
        resp = await _http_client.post(
            f"{GRADIUM_BASE_URL}/post/speech/tts",
            headers={"x-api-key": GRADIUM_API_KEY},
            json={
                "text": text,
                "voice_id": voice_id,
                "output_format": "wav",
                "only_audio": True,
            },
        )
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode()
    except Exception as e:
        logger.warning("TTS error: %s", e)
    return None


async def transcribe(audio_bytes: bytes, content_type: str) -> Optional[str]:
    if not GRADIUM_API_KEY:
        raise ValueError("GRADIUM_API_KEY is not set")
    try:
        resp = await _http_client.post(
            f"{GRADIUM_BASE_URL}/post/speech/asr",
            headers={"x-api-key": GRADIUM_API_KEY, "Content-Type": content_type},
            content=audio_bytes,
        )
        if resp.status_code != 200:
            logger.warning("STT status=%s body=%s", resp.status_code, resp.text[:400])
            return None

        # Gradium ASR returns NDJSON — one JSON object per line
        parts = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                msg_type = obj.get("type", "text")
                if msg_type == "error":
                    logger.warning("STT API error: %s", obj)
                    continue
                if msg_type == "end_text":
                    continue
                t = obj.get("text") or obj.get("transcript") or ""
                if t:
                    parts.append(str(t))
            except json.JSONDecodeError:
                pass

        result = " ".join(parts).strip()
        return result if result else None
    except Exception as e:
        logger.warning("STT error: %s", e)
    return None


# Slide chrome that never depends on the offer text — id, type, icon, title, accent
# colour. Kept server-side so the model only has to emit the variable fields
# (amounts, context, details). See PERFORMANCE.md → "Slides: server-side template".
_SLIDE_CHROME = [
    {
        "id": 1,
        "type": "intro",
        "icon": "🎉",
        "title": "We'd Like to Make You an Offer",
        "accent": "#6366f1",
    },
    {
        "id": 2,
        "type": "salary",
        "icon": "💰",
        "title": "Base Salary",
        "accent": "#0ea5e9",
    },
    {
        "id": 3,
        "type": "equity",
        "icon": "📈",
        "title": "Equity Package",
        "accent": "#22c55e",
    },
    {
        "id": 4,
        "type": "benefits",
        "icon": "✨",
        "title": "Benefits & Perks",
        "accent": "#f59e0b",
    },
    {
        "id": 5,
        "type": "total",
        "icon": "🏆",
        "title": "Total Compensation",
        "accent": "#ec4899",
    },
]


def _materialize_slides(data: dict) -> dict:
    """Merge the model's variable-field output with the fixed slide chrome."""
    return {
        "role_title": data.get("role_title", ""),
        "slides": [
            {
                **_SLIDE_CHROME[0],
                "role": data.get("role_title", ""),
                "tagline": data.get("intro_tagline", ""),
            },
            {
                **_SLIDE_CHROME[1],
                "amount": data.get("salary_amount", ""),
                "context": data.get("salary_context", ""),
                "details": data.get("salary_details", []),
            },
            {
                **_SLIDE_CHROME[2],
                "amount": data.get("equity_amount", ""),
                "context": data.get("equity_context", ""),
                "details": data.get("equity_details", []),
            },
            {**_SLIDE_CHROME[3], "items": data.get("benefits_items", [])},
            {
                **_SLIDE_CHROME[4],
                "amount": data.get("total_amount", ""),
                "context": "Year 1 total compensation",
                "breakdown": data.get("total_breakdown", []),
            },
        ],
    }


async def generate_slides(job_description: str, opening_offer: str) -> Optional[dict]:
    """Generate offer slide cards. Runs concurrently with TTS during /api/start.

    Only asks the model for the data-dependent fields; the chrome (id, type, icon,
    title, accent) is merged in by `_materialize_slides`. This cuts ~40% of the
    output tokens vs. asking the model to regenerate the entire structure, which
    directly cuts latency since output token generation dominates LLM time.
    """
    try:
        prompt = f"""Extract data from this job offer to fill 5 slide cards.
The numbers MUST match the offer text precisely.

OPENING OFFER MADE:
{opening_offer}

JOB DESCRIPTION:
{job_description}

Respond with JSON only:
{{
  "role_title": "<job title>",
  "intro_tagline": "<one compelling line>",
  "salary_amount": "<exact salary from offer>",
  "salary_context": "<market positioning>",
  "salary_details": ["<detail 1>", "<detail 2>"],
  "equity_amount": "<equity value from offer>",
  "equity_context": "<vesting info>",
  "equity_details": ["<detail 1>", "<detail 2>"],
  "benefits_items": ["<benefit 1>", "<benefit 2>", "<benefit 3>", "<benefit 4>", "<benefit 5>"],
  "total_amount": "<total comp year 1>",
  "total_breakdown": [
    {{"label": "Base Salary", "value": "<salary>"}},
    {{"label": "Target Bonus", "value": "<bonus>"}},
    {{"label": "Equity (Year 1)", "value": "<equity/4>"}},
    {{"label": "Benefits Value", "value": "~$20,000"}}
  ]
}}"""
        response = await _anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,  # was 1000 — smaller output now that the chrome is server-side
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "{}")
        start = text.find("{")
        end = text.rfind("}") + 1
        return _materialize_slides(json.loads(text[start:end]))
    except Exception as e:
        logger.warning("Slides error: %s", e)
        return None


class StartRequest(BaseModel):
    job_description: str
    difficulty: str
    voice_id: str


class ChatRequest(BaseModel):
    messages: list
    job_description: str
    difficulty: str
    voice_id: str


class TranscribeRequest(BaseModel):
    audio_b64: str
    content_type: str = "audio/wav"


class AssessRequest(BaseModel):
    messages: list
    job_description: str
    difficulty: str


class AssessResponse(BaseModel):
    score: int
    headline: str
    strengths: list[str]
    improvements: list[str]
    final_offer: str
    verdict: str


@app.get("/")
async def index(request: Request):
    # 304 Not Modified when the browser already has the current build.
    if request.headers.get("if-none-match") == _index_etag:
        return Response(status_code=304, headers={"ETag": _index_etag})
    return HTMLResponse(
        _index_html,
        headers={
            "ETag": _index_etag,
            # must-revalidate keeps the browser fast (re-uses local copy) but always
            # validates against the server, so a redeploy is picked up immediately.
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/config")
async def config():
    return JSONResponse(
        {
            "voices": VOICES,
            "difficulties": [
                {"id": k, "label": v["label"], "emoji": v["emoji"], "color": v["color"]}
                for k, v in DIFFICULTIES.items()
            ],
        },
        # Static config — let the browser cache it across page loads.
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/start")
async def start(req: StartRequest):
    voice = next((v for v in VOICES if v["id"] == req.voice_id), VOICES[0])
    if req.difficulty not in DIFFICULTIES:
        raise HTTPException(400, "Invalid difficulty")

    system = build_system_prompt(req.job_description, req.difficulty, voice)

    try:
        response = await _anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral", "ttl": "1h"},
            system=system,
            messages=[{"role": "user", "content": "Please make your opening offer."}],
        )
    except Exception as e:
        logger.error("Claude API error in /api/start: %s", e)
        raise HTTPException(502, f"AI service error: {e}")

    text = next((b.text for b in response.content if hasattr(b, "text")), "")

    # TTS and slides run concurrently — slides use the actual offer text.
    audio_b64, slides = await asyncio.gather(
        tts(text, voice["id"]),
        generate_slides(req.job_description, text),
    )

    return {"text": text, "audio_b64": audio_b64, "slides": slides}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    voice = next((v for v in VOICES if v["id"] == req.voice_id), VOICES[0])
    if req.difficulty not in DIFFICULTIES:
        raise HTTPException(400, "Invalid difficulty")

    system = build_system_prompt(req.job_description, req.difficulty, voice)

    claude_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in req.messages
        if m["role"] in ("user", "assistant")
    ]

    try:
        response = await _anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral", "ttl": "1h"},
            system=system,
            messages=claude_messages,
        )
    except Exception as e:
        logger.error("Claude API error in /api/chat: %s", e)
        raise HTTPException(502, f"AI service error: {e}")

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    audio_b64 = await tts(text, voice["id"])

    return {"text": text, "audio_b64": audio_b64}


@app.post("/api/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    audio_bytes = base64.b64decode(req.audio_b64)
    try:
        text = await transcribe(audio_bytes, req.content_type)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.warning("transcribe error: %s", e)
        raise HTTPException(500, f"Transcription error: {e}")
    if text is None:
        raise HTTPException(
            500, "No speech detected — try speaking louder or closer to the mic"
        )
    return {"text": text}


@app.post("/api/assess")
async def assess(req: AssessRequest):
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in req.messages
        if m["role"] in ("user", "assistant") and m.get("content")
    )
    # Schema-constrained output (output_config.format) replaces the fragile
    # text.find("{") parsing the previous version did. The API guarantees
    # valid JSON matching AssessResponse, and the compiled schema is cached
    # for 24h on the Anthropic side so repeat calls skip schema compilation.
    # See PERFORMANCE.md → "Structured outputs for /api/assess".
    prompt = f"""Evaluate this salary negotiation session.

JOB DESCRIPTION:
{req.job_description}

DIFFICULTY: {req.difficulty}

CONVERSATION:
{conversation}

Provide:
- score: 0-100 overall negotiation quality
- headline: one punchy sentence summarizing performance
- strengths: 3 things the candidate did well
- improvements: 3 things the candidate could improve
- final_offer: the AI's final stated offer (or "Unknown" if unclear)
- verdict: 2-3 sentences of coaching feedback"""

    try:
        response = await _anthropic_client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            output_format=AssessResponse,
        )
        return response.parsed_output
    except Exception as e:
        logger.warning("assess failed: %s", e)
        return AssessResponse(
            score=0,
            headline="Could not assess",
            strengths=[],
            improvements=[],
            final_offer="Unknown",
            verdict="Assessment failed. Please try again.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
