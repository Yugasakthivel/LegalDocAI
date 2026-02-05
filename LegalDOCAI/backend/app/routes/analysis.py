from fastapi import APIRouter, HTTPException, Form
from backend.app.database import documents_collection as collection, db
from backend.app.core.deps import get_openai_client
from backend.app.core.config import OPENAI_MODEL
from backend.app.services import analysis_service
import time
import json
import datetime

router = APIRouter(tags=["analysis"])

@router.post("/advanced")
async def analysis_advanced(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text","") if d else ""
    if not corpus:
        # Instead of 400 error, return empty valid structure to prevent frontend hanging
        return {
            "pipeline": [], 
            "summary": {
                "analytics": {"document_type": "Unknown", "legality_score": 0}, 
                "key_facts": {}, 
                "clause_risk": {"clauses": []}
            }
        }
        
    # Parallel execution using simple sequential calls for MVP stability
    # (Asyncio gather caused issues in some environments, keeping it simple/safe)
    
    # 1. Base NLP & Stats
    rs, an, working_text = analysis_service.analyze_text_overall(corpus)
    
    # 2. Key Facts (Critical)
    try:
        key_facts = analysis_service.extract_key_facts_llm(corpus)
    except:
        key_facts = {}

    # 3. Clauses (Critical)
    try:
        clause = analysis_service.clause_risk_detection(corpus)
    except:
        clause = {"clauses": []}
        
    # 4. Verification (Critical)
    try:
        v = analysis_service.ask_openai_for_verification_and_confidence(corpus)
    except:
        v = {"marker": "UNVERIFIED", "ai_confidence": 0}

    # 5. Optional/Background tasks (swallow errors)
    try: juris_ai = analysis_service.jurisdiction_specific_checks(corpus)
    except: juris_ai = {}
    
    try: juris_cur = analysis_service.curated_jurisdiction_templates(corpus)
    except: juris_cur = {}
    
    try: signers = analysis_service.validate_signers(corpus, rs.get("signers", []))
    except: signers = {}
    
    try: reg = analysis_service.analyze_registration(corpus)
    except: reg = {}
    
    try: mand = analysis_service.check_mandatory_fields(corpus)
    except: mand = {}
    
    try: humanized = analysis_service.humanize_text_ai(corpus)
    except: humanized = ""

    # Construct Response
    total = {
        "results_summary": rs,
        "analytics": {**an},
        "clause_risk": clause,
        "jurisdiction_ai": juris_ai,
        "jurisdiction_curated": juris_cur,
        "signer_validation": signers,
        "registration_analysis": reg,
        "verification_ai": v,
        "mandatory_fields": mand,
        "key_facts": key_facts,
        "humanized_text": humanized
    }
    
    # Merge into analytics for frontend compatibility
    total["analytics"]["clause_risk"] = clause
    total["analytics"]["verified_marker"] = v.get("marker")
    total["analytics"]["ai_confidence"] = v.get("ai_confidence")
    
    return {"pipeline": [], "summary": total}

@router.post("/explain")
async def analysis_explain(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    doc_type = None
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "analytics": 1})
        if d:
            corpus = d.get("combined_text", "") or ""
            an = d.get("analytics") or {}
            doc_type = an.get("document_type")
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        extra = f"Document type (from classifier): {doc_type}." if doc_type else ""
        prompt = (
            "Explain this legal document in simple English.\n"
            "Include:\n"
            "- What document it is\n"
            "- Legal importance\n"
            "- Whether it proves ownership or record only\n"
            "Use non-technical language.\n\n"
            + extra + "\n\nText:\n" + corpus[:6000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        explanation = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e)
        if "OPENAI_QUOTA_EXHAUSTED" in msg or "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            explanation = "Using local summarizer."
        else:
            explanation = ""
    if doc_id and explanation:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.explanation_simple": explanation}})
        except Exception:
            pass
    return {"explanation": explanation, "document_type": doc_type or analysis_service.classify_document_type(corpus)}

@router.post("/risk")
async def analysis_risk(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text","") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    result = analysis_service.llm_risk_validation(corpus)
    if doc_id:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.llm_risk": result}})
        except Exception:
            pass
    return {"risk": result}

@router.post("/feedback")
async def analysis_feedback(doc_id: str = Form(None), tag: str = Form(None), data: str = Form(None)):
    fb = db["training_feedback"]
    payload = {
        "doc_id": doc_id,
        "tag": tag,
        "data": json.loads(data) if data else None,
        "created_at": datetime.datetime.utcnow()
    }
    try:
        fb.insert_one(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")
    return {"status":"ok"}
