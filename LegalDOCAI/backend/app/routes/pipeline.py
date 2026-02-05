from fastapi import APIRouter, Form, HTTPException, Depends
from backend.app.services.ocr_service import extract_text
from backend.app.services.nlp_service import analyze_text
from backend.app.core.deps import get_current_user
from backend.app.services.risk_service import (
    clause_risk_detection,
    compute_combined_risk_index
)
from backend.app.services.compliance_service import (
    jurisdiction_checks,
    curated_checks,
    mandatory_fields
)
from backend.app.services.verification_service import verify_document, validate_signers
from backend.app.services.storage_service import save_document, get_document
from backend.app.vectorstore import add_document
from backend.app.services import analysis_service

router = APIRouter(tags=["pipeline"])

async def run_full_pipeline(doc_id: str):
    """
    FULL LEGALDOC PIPELINE
    (100% migrated from old main.py)
    """

    # 1️⃣ Load document from DB
    doc = get_document(doc_id)
    if not doc:
        return {"error": "Document not found"}

    text = doc["combined_text"]

    # 2️⃣ NLP Analysis
    results, analytics = analyze_text(text)

    clause_risk = clause_risk_detection(text)
    juris_ai = jurisdiction_checks(text)
    juris_curated = curated_checks(text)
    mand = mandatory_fields(text)
    verification = verify_document(text)
    llm_risk = analysis_service.llm_risk_validation(text)

    signers_found = results.get("signers", [])
    signer_validation = validate_signers(text, signers_found)

    # 8️⃣ Combine Scores
    analytics.update({
        "clause_risk": clause_risk,
        "jurisdiction_checks": juris_ai,
        "jurisdiction_curated": juris_curated,
        "mandatory_fields": mand,
        "verified_marker": verification.get("marker"),
        "ai_confidence": verification.get("ai_confidence"),
        "signer_validation": signer_validation,
        "llm_risk": llm_risk
    })

    analytics["legal_document_confidence"] = analytics.get("ai_confidence") or 0
    analytics["pages"] = (doc.get("pages") and len(doc["pages"])) or doc.get("total_pages") or 0
    analytics["combined_risk_index"] = compute_combined_risk_index(analytics)

    # 9️⃣ RAG Vector Indexing
    try:
        add_document({
            "doc_id": doc_id,
            "filename": doc.get("filename"),
            "combined_text": text
        })
    except Exception as e:
        print(f"RAG Indexing failed: {e}")

    # 🔟 Save back to Mongo
    try:
        save_document(doc_id, {
            "analytics": analytics,
            "pipeline.classify": {"document_type": analytics.get("document_type")},
            "pipeline.ner": {"entities": results, "stats": analytics},
            "pipeline.risk": {"risk": analytics.get("clause_risk")},
            "pipeline.timeline": {"timeline": analytics.get("timeline")},
        })
    except Exception:
        save_document(doc_id, {"analytics": analytics})

    return {
        "doc_id": doc_id,
        "results_summary": results,
        "analytics": analytics
    }

@router.post("/full")
async def full(doc_id: str = Form(...), user: dict = Depends(get_current_user)):
    res = await run_full_pipeline(doc_id)
    if res.get("error"):
        raise HTTPException(status_code=404, detail=res["error"])
    return res
