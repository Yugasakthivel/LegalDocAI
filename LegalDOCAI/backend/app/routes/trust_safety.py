import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.config import UPLOAD_FOLDER
from backend.app.services.trust_safety_service import (
    PIIDetectionService,
    FairnessAuditService,
    AccountabilityService,
)

router = APIRouter(prefix="/trust-safety", tags=["trust-safety"])


TRUST_DIR = os.path.join(UPLOAD_FOLDER, "trust_safety")
os.makedirs(TRUST_DIR, exist_ok=True)

_pii = PIIDetectionService()
_fair = FairnessAuditService()
_acct = AccountabilityService()


class PrivacyRequest(BaseModel):
    text: str
    mode: Optional[str] = None
    doc_id: Optional[str] = None


class FeedbackInput(BaseModel):
    model_config = {"protected_namespaces": ()}
    original_text: str = ""
    model_prediction: str = ""
    user_correction: str = ""
    model_confidence: float | None = None
    feedback_type: str = "correction"
    notes: str = ""


class ConfidenceInput(BaseModel):
    text: Optional[str] = None
    ml_confidence: Optional[float] = None
    mode: Optional[str] = None
    doc_id: Optional[str] = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _write_json(prefix: str, payload: Dict[str, Any]) -> str:
    name = f"{prefix}_{int(time.time() * 1000)}.json"
    path = os.path.join(TRUST_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return name


def _read_json_files(prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for fn in os.listdir(TRUST_DIR):
            if not fn.startswith(prefix) or not fn.endswith(".json"):
                continue
            fp = os.path.join(TRUST_DIR, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    except Exception:
        return []
    return out


def _detect_pii(text: str) -> List[Dict[str, Any]]:
    patterns = [
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
        ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b"),
    ]
    entities: List[Dict[str, Any]] = []
    for label, pat in patterns:
        for m in re.finditer(pat, text):
            entities.append(
                {
                    "text": m.group(0),
                    "label": label,
                    "start": m.start(),
                    "end": m.end(),
                    "confidence": 0.95 if label in ("SSN", "CREDIT_CARD") else 0.9,
                }
            )
    return sorted(entities, key=lambda x: x["start"])


def _privacy_score(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = {"SSN": 50, "CREDIT_CARD": 50, "EMAIL": 10, "PHONE": 10}
    baseline = 100
    deductions = []
    total = 0
    for e in entities:
        d = weights.get(e["label"], 1)
        total += d
        deductions.append({"type": e["label"], "value": -d, "text": e["text"]})
    score = max(0, baseline - total)
    if score >= 80:
        compliance = "EXCELLENT"
    elif score >= 60:
        compliance = "GOOD"
    elif score >= 40:
        compliance = "FAIR"
    elif score >= 20:
        compliance = "POOR"
    else:
        compliance = "CRITICAL"
    return {
        "score": score,
        "compliance_level": compliance,
        "breakdown": {"baseline": baseline, "deductions": deductions, "total_deduction": total},
    }


def _confidence_for_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"confidence_score": 0.0, "confidence_level": "VERY_LOW", "requires_review": True}
    legal_terms = [
        "agreement",
        "party",
        "liability",
        "termination",
        "confidential",
        "jurisdiction",
        "warranty",
        "indemnity",
    ]
    term_hits = sum(1 for t in legal_terms if t in raw.lower())
    term_score = min(1.0, term_hits / 6.0)
    length_score = min(1.0, len(raw) / 3000.0)
    noise_penalty = 0.0
    if re.search(r"[^\x09\x0A\x0D\x20-\x7E]", raw):
        noise_penalty += 0.07
    if raw.count("?") > 5:
        noise_penalty += 0.05
    confidence = max(0.0, min(1.0, 0.55 * term_score + 0.45 * length_score - noise_penalty))
    if confidence >= 0.9:
        level = "VERY_HIGH"
    elif confidence >= 0.8:
        level = "HIGH"
    elif confidence >= 0.7:
        level = "MEDIUM"
    elif confidence >= 0.6:
        level = "LOW"
    else:
        level = "VERY_LOW"
    return {"confidence_score": round(confidence, 4), "confidence_level": level, "requires_review": confidence < 0.7}


@router.get("/dashboard")
async def trust_dashboard():
    feedback = _read_json_files("feedback")
    privacy = _read_json_files("privacy")
    confidence = _read_json_files("confidence")
    avg_privacy = round(sum(float(x.get("score", 0)) for x in privacy) / len(privacy), 2) if privacy else 0.0
    avg_conf = round(sum(float(x.get("confidence_score", 0)) for x in confidence) / len(confidence), 4) if confidence else 0.0
    return {
        "privacy_protection": {
            "documents_processed": len(privacy),
            "average_privacy_score": avg_privacy,
        },
        "human_feedback": {
            "feedback_entries": len(feedback),
        },
        "confidence_scoring": {
            "documents_scored": len(confidence),
            "average_confidence": avg_conf,
            "low_confidence_count": len([x for x in confidence if float(x.get("confidence_score", 0)) < 0.7]),
        },
        "fairness_audits": {
            "completed_audits": len(_read_json_files("fairness")),
        },
    }


@router.post("/privacy/redact")
async def privacy_redact(payload: PrivacyRequest):
    text = payload.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    out = _pii.redact(text)
    return out


@router.post("/fairness/audit")
async def fairness_audit(mode: Optional[str] = "from_feedback"):
    out = _fair.run_audit(mode or "from_feedback")
    return out


@router.post("/confidence/score")
async def confidence_score(payload: ConfidenceInput):
    conf = _acct.confidence_score(text=payload.text, ml_confidence=payload.ml_confidence)
    out = {"text": (payload.text or "")[:1000], **conf, "timestamp": _now_iso()}
    _write_json("confidence", out)
    return out


@router.post("/feedback")
async def feedback_collect(payload: FeedbackInput):
    out = {
        "feedback_id": f"feedback_{int(time.time() * 1000)}",
        "original_text": payload.original_text,
        "model_prediction": payload.model_prediction,
        "user_correction": payload.user_correction,
        "model_confidence": payload.model_confidence,
        "feedback_type": payload.feedback_type,
        "notes": payload.notes,
        "timestamp": _now_iso(),
    }
    fname = _write_json("feedback", out)
    return {"success": True, "feedback_id": out["feedback_id"], "file": fname, "timestamp": out["timestamp"]}


@router.get("/audit-trails")
async def audit_trails():
    return {"events": _acct.list_audit_trails()}


@router.get("/feedback/summary")
async def feedback_summary():
    return _acct.summarize_feedback()


@router.get("/fairness/reports")
async def fairness_reports():
    return {"reports": _fair.list_reports()}
