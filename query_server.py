"""
query_server.py  --  D2: HTTP wrapper around query_tools
========================================================

Serves the six deterministic query tools (and the optional LLM chat endpoint)
over HTTP on port 8765 by default. This is the service that tree_viz.py,
query_ui.py, and any external client talk to.

Endpoints
---------
  GET  /health                         -> liveness + run dir + event count
  GET  /manifest                       -> raw manifest.json
  GET  /tools                          -> machine-readable tool schemas
  GET  /runs                           -> list available run dirs
  GET  /run_summary                    -> first/last t, schema version, counts
  GET  /jump_to_time?t=...             -> full event snapshot at/just-before t
  GET  /interval_stats?t1=..&t2=..&fields=a,b
  GET  /find_mode_changes?t1=..&t2=..&field=action.B_mode
  GET  /explain_change?t=..&window=5
  GET  /top_rules_at?t=..&k=5
  POST /chat   body {"question": "...", "max_steps": 6}   (needs ANTHROPIC_API_KEY)

Run
---
  cd G:\\yoran_rl
  G:\\yoran_rl\\.venv\\Scripts\\python.exe -m uvicorn query_server:app --host 127.0.0.1 --port 8765

Run-dir selection
-----------------
  * If env var RUN_DIR is set, that exact directory is used.
  * Otherwise the most recently modified subdirectory under ./runs is used.
  * GET /runs lists what's available; POST is not needed -- restart with a
    different RUN_DIR to switch, or call /reload to re-scan.

The LLM endpoint is optional. If the anthropic SDK or the API key is missing,
every other endpoint still works; only /chat returns an error.
"""

import os
import glob
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import query_tools as qt

# llm_summariser is optional. Import lazily so the server still boots if the
# anthropic SDK isn't installed.
try:
    import llm_summariser as llm
    _HAS_LLM = True
except Exception as _e:           # noqa: BLE001
    llm = None
    _HAS_LLM = False
    _LLM_IMPORT_ERROR = str(_e)


# ----------------------------------------------------------------------
# Run-dir resolution
# ----------------------------------------------------------------------

def _list_run_dirs(base: str = "runs") -> List[str]:
    if not os.path.isdir(base):
        return []
    subs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    # Only keep dirs that actually contain an events file or manifest.
    subs = [s for s in subs
            if os.path.exists(os.path.join(s, "events.jsonl"))
            or os.path.exists(os.path.join(s, "manifest.json"))]
    subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subs


def _pick_run_dir() -> str:
    env = os.environ.get("RUN_DIR")
    if env:
        if not os.path.isdir(env):
            raise RuntimeError(f"RUN_DIR={env} is not a directory.")
        return env
    subs = _list_run_dirs("runs")
    if not subs:
        raise RuntimeError(
            "No run directories found under ./runs. "
            "Start logger_harmoniser.py first, or set RUN_DIR."
        )
    return subs[0]


# Resolve at import time so STORE is ready. If it fails, defer the error to
# the first request rather than killing uvicorn import.
try:
    RUN_DIR = _pick_run_dir()
    STORE = qt.EventStore(RUN_DIR)
    _STARTUP_ERROR = None
except Exception as e:            # noqa: BLE001
    RUN_DIR = None
    STORE = None
    _STARTUP_ERROR = str(e)


app = FastAPI(title="MARL+LMUT Query Server", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_store():
    if STORE is None:
        raise HTTPException(
            status_code=503,
            detail=f"Query store not ready: {_STARTUP_ERROR}. "
                   f"Start logger_harmoniser.py or set RUN_DIR, then call /reload.",
        )


# ----------------------------------------------------------------------
# Meta endpoints
# ----------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "MARL+LMUT Query Server",
        "version": "1.1.0",
        "run_dir": RUN_DIR,
        "llm_enabled": _HAS_LLM,
        "endpoints": [
            "/health", "/manifest", "/tools", "/runs", "/run_summary",
            "/jump_to_time", "/interval_stats", "/find_mode_changes",
            "/explain_change", "/top_rules_at", "/chat", "/reload", "/docs",
        ],
    }


@app.get("/health")
def health():
    if STORE is None:
        return {"ok": False, "error": _STARTUP_ERROR, "run_dir": None, "n_events": 0}
    STORE.reload_if_changed()
    return {"ok": True, "run_dir": RUN_DIR, "n_events": STORE.count(),
            "llm_enabled": _HAS_LLM}


@app.get("/runs")
def runs():
    return {"run_dirs": _list_run_dirs("runs"), "active": RUN_DIR}


@app.post("/reload")
def reload():
    """Re-scan ./runs and point STORE at the newest run dir."""
    global RUN_DIR, STORE, _STARTUP_ERROR
    try:
        RUN_DIR = _pick_run_dir()
        STORE = qt.EventStore(RUN_DIR)
        _STARTUP_ERROR = None
        STORE.reload_if_changed()
        return {"ok": True, "run_dir": RUN_DIR, "n_events": STORE.count()}
    except Exception as e:        # noqa: BLE001
        _STARTUP_ERROR = str(e)
        STORE = None
        RUN_DIR = None
        raise HTTPException(status_code=503, detail=_STARTUP_ERROR)


@app.get("/manifest")
def manifest():
    _require_store()
    return STORE.manifest()


@app.get("/tools")
def tools():
    return {"tools": qt.TOOL_SCHEMAS}


# ----------------------------------------------------------------------
# Deterministic query endpoints
# ----------------------------------------------------------------------

@app.get("/run_summary")
def run_summary():
    _require_store()
    return qt.run_summary(STORE)


@app.get("/jump_to_time")
def jump_to_time(t: float):
    _require_store()
    return qt.jump_to_time(STORE, t)


@app.get("/interval_stats")
def interval_stats(t1: float, t2: float, fields: Optional[str] = None):
    _require_store()
    flist = None
    if fields:
        flist = [s.strip() for s in fields.split(",") if s.strip()]
    return qt.interval_stats(STORE, t1, t2, flist)


@app.get("/find_mode_changes")
def find_mode_changes(t1: float, t2: float, field: str = "action.B_mode"):
    _require_store()
    return qt.find_mode_changes(STORE, t1, t2, field)


@app.get("/explain_change")
def explain_change(t: float, window: int = 5):
    _require_store()
    return qt.explain_change(STORE, t, window)


@app.get("/top_rules_at")
def top_rules_at(t: float, k: int = 5):
    _require_store()
    return qt.top_rules_at(STORE, t, k)


# ----------------------------------------------------------------------
# LLM chat endpoint (optional)
# ----------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    max_steps: int = 6


@app.post("/chat")
def chat(req: ChatRequest):
    _require_store()
    if not _HAS_LLM:
        raise HTTPException(
            status_code=501,
            detail=f"LLM endpoint unavailable: {_LLM_IMPORT_ERROR}. "
                   f"Install the anthropic SDK and set ANTHROPIC_API_KEY.",
        )
    try:
        answer, trace = llm.answer_question(STORE, req.question, max_steps=req.max_steps)
        return {"ok": True, "answer": answer, "tool_calls": trace}
    except Exception as e:        # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
