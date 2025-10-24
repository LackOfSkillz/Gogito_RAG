import os, sys, subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from cogito_query import ask as rag_ask, ask_stream as rag_ask_stream
from cogito_cache import get_cache

LM_STUDIO_URL = "http://localhost:1234"

app = FastAPI(title="Cogito API", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    question: str
    depth: int = 8
    temperature: float = 0.3
    max_tokens: int = 768
    model: Optional[str] = None
    mode: str = "balanced"  # "fast" | "balanced" | "thorough"
    use_cache: bool = True

@app.get("/api/models")
def models():
    try:
        r = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=3)
        r.raise_for_status()
        data = r.json()
        ids = [m["id"] for m in data.get("data", []) if "id" in m]
        # optional: hide embedding models
        ids = [i for i in ids if "embed" not in i.lower() and "embedding" not in i.lower()]
        return {"models": ids}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.post("/api/ask")
def ask(payload: AskReq):
    return rag_ask(
        payload.question.strip(),
        depth=payload.depth,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        model=payload.model,
        mode=payload.mode,
        use_cache=payload.use_cache,
    )

@app.post("/api/ask_stream")
def ask_stream(payload: AskReq):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Returns tokens as they're generated.
    """
    def event_generator():
        try:
            for chunk in rag_ask_stream(
                payload.question.strip(),
                depth=payload.depth,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                model=payload.model,
                mode=payload.mode,
            ):
                # SSE format: data: {json}\n\n
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Signal completion
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@app.get("/api/cache/stats")
def cache_stats():
    """Get cache statistics."""
    try:
        cache = get_cache()
        return cache.get_stats()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/cache/clear")
def cache_clear():
    """Clear the entire cache."""
    try:
        cache = get_cache()
        cache.clear_all()
        return {"ok": True, "message": "Cache cleared"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/ingest")
def ingest():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-u", str(ROOT / "cogito_loader.py"), "--progress"]
    try:
        out = subprocess.check_output(cmd, cwd=str(ROOT), env=env, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        return {"ok": True, "log": out[-5000:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "log": (e.output or "")[-5000:], "code": e.returncode}
