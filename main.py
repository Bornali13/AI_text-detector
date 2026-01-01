# main.py — FastAPI backend for your app + Sapling external AI detection

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re
import os
import requests
from datetime import datetime
from typing import Optional, Any, Dict, List

app = FastAPI(title="AI Disclosure — MVP API (Sapling)")

# ----------------------------
# CORS (avoid "*" with credentials)
# ----------------------------
ALLOWED_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,  # OK because origins are explicit
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# External AI Detector (Sapling)
# ----------------------------
SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"

def sapling_detect(text: str) -> Dict[str, Any]:
    """
    Calls Sapling AI-detection API and returns:
      {
        "ai_percentage": float,   # 0..100
        "raw_score": float,       # 0..1
        "note": str|None
      }

    Notes:
    - This is a probabilistic signal, not definitive proof.
    - We cap text length to reduce costs / avoid limits.
    """
    key = os.getenv("SAPLING_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="SAPLING_API_KEY is not set (server config).")

    doc = (text or "").strip()
    if len(doc) < 50:
        return {"ai_percentage": 0.0, "raw_score": 0.0, "note": "Text too short"}

    payload = {
        "key": key,
        "text": doc[:5000],  # safe cap; adjust if your plan allows more
    }

    try:
        r = requests.post(SAPLING_URL, json=payload, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sapling request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Sapling error {r.status_code}: {r.text}")

    data = r.json()

    # Sapling typically returns "score" in [0,1]
    score = float(data.get("score", 0.0))
    pct = round(score * 100.0, 2)

    return {"ai_percentage": pct, "raw_score": score, "note": None}

# ----------------------------
# Simple metrics (same logic as your student.js)
# ----------------------------
def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            c = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[m][n]

def uniq_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z']+", t.lower()) if len(w) > 2}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union

def split_sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", t.strip()) if s.strip()]

def structure_shift(draft: str, final: str) -> float:
    pd = draft.count("\n\n") + 1
    pf = final.count("\n\n") + 1
    return min(1.0, abs(pd - pf) / 10.0)

def compute_metrics(draft: str, final: str) -> dict:
    ed = levenshtein(draft, final)
    edn = ed / max(len(draft), len(final), 1)
    lex = 1 - jaccard(uniq_words(draft), uniq_words(final))
    sd, sf = split_sentences(draft), split_sentences(final)
    reused = sum(1 for s in sd if s in sf)
    reuse_rate = reused / max(len(sd), 1)
    paraphrase = max(0.0, 1 - reuse_rate - 0.2)
    struct = structure_shift(draft, final)
    improvement = 100.0 * (0.25*edn + 0.20*paraphrase + 0.15*struct + 0.10*0.5 + 0.10*lex)
    return dict(
        edit_distance_norm=edn,
        lexical_change=lex,
        sentence_reuse_rate=reuse_rate,
        paraphrase_rate=paraphrase,
        structure_shift=struct,
        improvement_score=improvement
    )

# ----------------------------
# In-memory storage for demo (resets on server restart)
# ----------------------------
SUBMISSIONS: List[Dict[str, Any]] = []
NEXT_ID = 1

@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------
# Compare (no saving)
# ----------------------------
class CompareIn(BaseModel):
    draft_text: str = ""
    final_text: str

class CompareOut(BaseModel):
    edit_distance_norm: float
    lexical_change: float
    sentence_reuse_rate: float
    paraphrase_rate: float
    structure_shift: float
    improvement_score: float
    ai_percentage: Optional[float] = None

@app.post("/api/compare", response_model=CompareOut)
def api_compare(body: CompareIn):
    m = compute_metrics(body.draft_text or "", body.final_text)

    # Best-effort external detection (don’t break compare if API fails)
    try:
        ai = sapling_detect(body.final_text)
        m["ai_percentage"] = ai["ai_percentage"]
    except HTTPException:
        m["ai_percentage"] = None

    return m

# ----------------------------
# External AI detect endpoint (frontend can call)
# ----------------------------
class DetectIn(BaseModel):
    text: str = Field(..., min_length=1)

class DetectOut(BaseModel):
    ai_percentage: float
    raw_score: float
    note: Optional[str] = None

@app.post("/api/ai-detect", response_model=DetectOut)
def api_ai_detect(body: DetectIn):
    ai = sapling_detect(body.text)
    return ai

# ----------------------------
# Submit (creates record + metrics + AI)
# ----------------------------
class SubmissionIn(BaseModel):
    student_name: str
    student_id: str
    student_email: Optional[str] = None
    course_code: str
    assignment_id: str
    used_ai: bool = False
    used_rewrite: bool = False
    used_research: bool = False
    used_complete: bool = False
    evidence_text: Optional[str] = None
    draft_text: Optional[str] = ""
    final_text: str

class SubmissionOut(BaseModel):
    id: int
    checks_used: int
    locked: bool
    metrics: CompareOut
    ai_raw_score: Optional[float] = None

@app.post("/api/submit", response_model=SubmissionOut)
def api_submit(body: SubmissionIn):
    global NEXT_ID
    metrics = compute_metrics(body.draft_text or "", body.final_text)

    ai_raw_score = None
    try:
        ai = sapling_detect(body.final_text)
        metrics["ai_percentage"] = ai["ai_percentage"]
        ai_raw_score = ai["raw_score"]
    except HTTPException:
        metrics["ai_percentage"] = None

    record = {
        "id": NEXT_ID,
        "ts": datetime.utcnow().isoformat(),
        "checks_used": 1,
        "locked": False,
        **body.model_dump(),
        "metrics": metrics,
        "ai_raw_score": ai_raw_score,
    }
    SUBMISSIONS.append(record)
    NEXT_ID += 1

    return {
        "id": record["id"],
        "checks_used": record["checks_used"],
        "locked": record["locked"],
        "metrics": record["metrics"],
        "ai_raw_score": record["ai_raw_score"],
    }

# ----------------------------
# Another check (up to 3)
# ----------------------------
@app.post("/api/check/{submission_id}", response_model=CompareOut)
def api_check(submission_id: int):
    sub = next((s for s in SUBMISSIONS if s["id"] == submission_id), None)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if sub["locked"]:
        raise HTTPException(status_code=400, detail="Finalized; checks locked")
    if sub["checks_used"] >= 3:
        raise HTTPException(status_code=400, detail="No checks remaining")

    m = compute_metrics(sub.get("draft_text") or "", sub["final_text"])

    # refresh AI score best-effort
    try:
        ai = sapling_detect(sub["final_text"])
        m["ai_percentage"] = ai["ai_percentage"]
        sub["ai_raw_score"] = ai["raw_score"]
    except HTTPException:
        m["ai_percentage"] = sub.get("metrics", {}).get("ai_percentage")

    sub["metrics"] = m
    sub["checks_used"] += 1
    return m

# ----------------------------
# Finalize locks the record
# ----------------------------
@app.post("/api/finalize/{submission_id}")
def api_finalize(submission_id: int):
    sub = next((s for s in SUBMISSIONS if s["id"] == submission_id), None)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    sub["locked"] = True
    return {"ok": True}

# ----------------------------
# Teacher list (for teacher page)
# ----------------------------
@app.get("/api/teacher/submissions")
def teacher_list(q: Optional[str] = None, course: Optional[str] = None, sort: str = "tsDESC"):
    data = SUBMISSIONS[:]

    if q:
        ql = q.lower()
        data = [
            s for s in data
            if any(ql in (s.get(k, "") or "").lower()
                   for k in ("student_name", "student_id", "course_code", "assignment_id", "student_email"))
        ]

    if course:
        data = [s for s in data if (s.get("course_code") or "").strip() == course]

    if sort == "scoreDESC":
        data.sort(key=lambda s: s["metrics"]["improvement_score"], reverse=True)
    elif sort == "scoreASC":
        data.sort(key=lambda s: s["metrics"]["improvement_score"])
    else:
        data.sort(key=lambda s: s["ts"], reverse=True)

    return data

