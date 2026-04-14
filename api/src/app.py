"""
FastAPI application — Forging Line Delay Diagnostics API.
Reference data is loaded once at startup.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from diagnose import diagnose

# ── Startup: load reference times once ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_FILE = BASE_DIR / "reference_times.json"

with open(REFERENCE_FILE) as f:
    REFERENCE_TIMES = json.load(f)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Forging Line Delay Diagnostics API",
    description="Receives timing data for a forging piece and returns a delay diagnosis.",
    version="1.0.0",
)


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return HTTP 400 with {"error": "..."} for invalid request bodies."""
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "invalid request body"
    return JSONResponse(status_code=400, content={"error": msg})


# ── Request schema ────────────────────────────────────────────────────────────
class PieceRequest(BaseModel):
    piece_id: str
    die_matrix: int
    lifetime_2nd_strike_s: Optional[float] = None
    lifetime_3rd_strike_s: Optional[float] = None
    lifetime_4th_strike_s: Optional[float] = None
    lifetime_auxiliary_press_s: Optional[float] = None
    lifetime_bath_s: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/diagnose")
def diagnose_piece(request: PieceRequest):
    """Diagnose a single forging piece for segment delays."""
    try:
        result = diagnose(request.model_dump(), REFERENCE_TIMES)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
