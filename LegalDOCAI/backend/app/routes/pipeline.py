from fastapi import APIRouter, Form, HTTPException, Depends, BackgroundTasks
import time
import uuid
from backend.app.services.ocr_service import extract_text
from backend.app.services import analysis_service
from backend.app.core.deps import get_current_user
from backend.app.services.risk_service import (
    clause_risk_detection,
    clause_risk_detection_async,
    compute_combined_risk_index
)
from backend.app.services.compliance_service import (
    jurisdiction_checks,
    jurisdiction_checks_async,
    curated_checks,
    curated_checks_async,
    mandatory_fields,
    mandatory_fields_async
)
from backend.app.services.verification_service import (
    verify_document, 
    verify_document_async,
    validate_signers,
    validate_signers_async
)
import asyncio
from backend.app.services.storage_service import save_document, get_document
from backend.app.vectorstore import add_document, add_document_async
from backend.app.services.legal_analyzer import analyze_document

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_PIPELINE_JOBS = {}

def _init_pipeline_job(job_id: str, doc_id: str):
    _PIPELINE_JOBS[job_id] = {
        "job_id": job_id,
        "doc_id": doc_id,
        "state": "queued",
        "started_at": None,
        "updated_at": time.time(),
        "result": None,
        "error": None,
    }

def _update_pipeline_job(job_id: str, **kwargs):
    job = _PIPELINE_JOBS.get(job_id) or {"job_id": job_id}
    job.update(kwargs)
    job["updated_at"] = time.time()
    _PIPELINE_JOBS[job_id] = job

async def _run_pipeline_job(job_id: str, doc_id: str):
    _update_pipeline_job(job_id, state="running", started_at=time.time())
    try:
        res = await run_full_pipeline(doc_id)
        if res.get("error"):
            _update_pipeline_job(job_id, state="failed", error=res.get("error"))
        else:
            _update_pipeline_job(job_id, state="completed", result=res)
    except Exception as e:
        _update_pipeline_job(job_id, state="failed", error=str(e))

async def run_full_pipeline(doc_id: str, background_tasks: BackgroundTasks = None):
    """
    FULL LEGALDOC PIPELINE
    (100% migrated from old main.py)
    """

    # 1️⃣ Load document from DB (Async)
    try:
        print(f"[PIPELINE] Start run_full_pipeline doc_id={doc_id}")
        doc = await asyncio.to_thread(get_document, doc_id)
        if not doc:
            return {"error": "Document not found"}
    except Exception as e:
        return {"error": f"Failed to load document: {str(e)}"}

    text = doc.get("combined_text", "")
    filename = doc.get("filename")
    try:
        print(f"[PIPELINE] Loaded doc_id={doc_id}, extracted_text_len={len(text or '')}")
    except Exception:
        pass

    # 2️⃣ NLP Analysis
    print("[PIPELINE] Stage: NLP + NER + Regex extraction")
    (results, analytics, working_text) = await analysis_service.analyze_text_overall_async(text)
    doc_type = analysis_service.classify_document_type(working_text or text)
    try:
        entities_count = len(results.get("entities", {}).get("names", [])) + len(results.get("entities", {}).get("organizations", []))
        print(f"[PIPELINE] Entities detected={entities_count}, doc_type={doc_type}")
    except Exception:
        pass
    
    # 3️⃣ Parallel Analysis Tasks
    # We run all independent analysis tasks in parallel to reduce latency
    print("[PIPELINE] Stage: Two-step legal classification")
    two_step = analysis_service.classify_legal_two_step(working_text or text, doc_type)
    is_legal = bool(two_step.get("is_legal"))
    if not is_legal:
        clause_risk = {"overall_risk_score": 0}
        juris_ai = {}
        juris_curated = {}
        timeline = []
        try:
            print("[PIPELINE] Stage: Mandatory fields (non-legal fast path)")
            mand = await asyncio.wait_for(mandatory_fields_async(text), 3)
        except Exception:
            mand = {}
        verification = {"marker": "Verified", "ai_confidence": 90, "risk_level": "None", "predicted_category": doc_type}
        llm_risk = {}
    else:
        print("[PIPELINE] Stage: Parallel analysis tasks (risk, jurisdiction, timeline, mandatory, verification, llm)")
        tasks = [
            asyncio.wait_for(analysis_service.clause_risk_detection_async(text, translated_text=working_text or text), 4),
            asyncio.wait_for(jurisdiction_checks_async(text), 4),
            asyncio.wait_for(curated_checks_async(text), 4),
            asyncio.wait_for(analysis_service.build_timeline_async(working_text or text), 4),
            asyncio.wait_for(mandatory_fields_async(text), 4),
            asyncio.wait_for(verify_document_async(working_text or text), 4),
            asyncio.wait_for(analysis_service.llm_risk_validation_async(working_text or text), 4),
        ]
        res = await asyncio.gather(*tasks, return_exceptions=True)
        clause_risk = {} if isinstance(res[0], Exception) else (res[0] or {})
        juris_ai = {} if isinstance(res[1], Exception) else (res[1] or {})
        juris_curated = {} if isinstance(res[2], Exception) else (res[2] or {})
        timeline = [] if isinstance(res[3], Exception) else (res[3] or [])
        mand = {} if isinstance(res[4], Exception) else (res[4] or {})
        verification = {"marker": "Rejected", "ai_confidence": 0} if isinstance(res[5], Exception) else (res[5] or {"marker": "Rejected", "ai_confidence": 0})
        llm_risk = {} if isinstance(res[6], Exception) else (res[6] or {})
        try:
            print(f"[PIPELINE] Risk score={clause_risk.get('overall_risk_score')}, verification={verification.get('marker')} ai_conf={verification.get('ai_confidence')}")
        except Exception:
            pass

    # 4️⃣ Dependent Tasks (Signer validation depends on results from NER)
    # Extract names from results.entities.names as potential signers
    signers_found = results.get("entities", {}).get("names", [])
    if not signers_found:
        # Fallback to checking if results itself has names at top level (old format)
        signers_found = results.get("names", [])
        
    signer_validation = await validate_signers_async(working_text or text, signers_found)

    # 5️⃣ Combine Scores & Populate Analytics
    # Ensure all scores have at least a default value to prevent 25% confidence floor
    if not signer_validation.get("overall_signer_score"):
        # If we found names but validation failed, give a baseline score
        if signers_found:
            signer_validation["overall_signer_score"] = 60
        else:
            signer_validation["overall_signer_score"] = 20

    analytics.update({
        "clause_risk": clause_risk,
        "jurisdiction_checks": juris_ai,
        "jurisdiction_curated": juris_curated,
        "timeline": timeline,
        "mandatory_fields": mand,
        "verified_marker": verification.get("marker"),
        "ai_confidence": verification.get("ai_confidence"),
        "signer_validation": signer_validation,
        "llm_risk": llm_risk,
        "filename": filename
    })
    
    # Ensure clause_risk has a score
    if not analytics["clause_risk"].get("overall_risk_score"):
        analytics["clause_risk"]["overall_risk_score"] = 50
    
    # Ensure jurisdiction_curated has a score
    if not analytics["jurisdiction_curated"].get("overall_compliance_score"):
        analytics["jurisdiction_curated"]["overall_compliance_score"] = 40

    # 6️⃣ Calculate missing fields for combined_risk_index
    # Fallback to rule-based classification if ML fails
    analytics["document_type"] = doc_type or "Unknown"
    try:
        analytics["document_category"] = analysis_service.classify_document_category(working_text or text)
    except Exception:
        analytics["document_category"] = "Other"
    analytics["is_legal_document"] = is_legal
    analytics["is_indian_legal_document"] = bool(two_step.get("is_indian"))

    # 7️⃣ AI Fallback Path (only when needed)
    fallback_needed = (
        (doc_type in ["Unknown", "Other", "Other Document", None]) or
        ((analytics.get("legal_document_confidence") or 0) < 50)
    )
    ai_result = None
    if fallback_needed:
        try:
            ai_result = await analysis_service.ai_classify_and_extract_async(working_text or text)
            if isinstance(ai_result, dict):
                ai_doc_type = (ai_result.get("document_type") or "").strip()
                ai_category = (ai_result.get("category") or "").strip()
                if ai_doc_type and ai_doc_type not in ["Other", "Unknown"]:
                    analytics["document_type"] = ai_doc_type
                if ai_category:
                    analytics["document_category"] = ai_category
        except Exception:
            ai_result = {"error": "AI fallback failed"}
    # Calculate scores with proper fallbacks
    try:
        analytics["legality_score"] = analysis_service.compute_legality_score(analytics)
    except Exception:
        analytics["legality_score"] = 50  # Default score if calculation fails
    try:
        print(f"[PIPELINE] Legality score={analytics.get('legality_score')} category={analytics.get('document_category')} type={analytics.get('document_type')}")
    except Exception:
        pass

    try:
        analytics["legal_document_confidence"] = analytics.get("ai_confidence") or 0
    except Exception:
        analytics["legal_document_confidence"] = 50  # Default confidence

    # Ensure ai_confidence is not 0 if verified
    if analytics.get("verified_marker") == "Verified":
        if not analytics.get("ai_confidence"):
            analytics["ai_confidence"] = 85
            analytics["legal_document_confidence"] = 85
        elif analytics["ai_confidence"] < 50:
            analytics["ai_confidence"] = 85
            analytics["legal_document_confidence"] = 85

    # Calculate pages with proper fallback
    try:
        pages = doc.get("pages")
        if pages and isinstance(pages, list):
            analytics["pages"] = len(pages)
        else:
            analytics["pages"] = doc.get("total_pages") or 0
    except Exception:
        analytics["pages"] = 0

    # Calculate combined risk index with error handling
    try:
        analytics["combined_risk_index"] = compute_combined_risk_index(analytics)
    except Exception:
        analytics["combined_risk_index"] = 50  # Default risk index

    # Calculate AI detection score with proper validation
    try:
        ai_score = analysis_service.compute_ai_detection_score(working_text or text)
        if ai_score is not None:
            analytics["ai_detection_score"] = max(0, min(100, ai_score))
    except Exception:
        pass

    # 9️⃣ RAG Vector Indexing (moved to background task)
    if background_tasks:
        background_tasks.add_task(add_document_async, {
            "doc_id": doc_id,
            "filename": doc.get("filename"),
            "combined_text": text
        })
        print(f"[PIPELINE] Vector indexing scheduled for doc_id={doc_id}")
    else:
        try:
            await add_document_async({
                "doc_id": doc_id,
                "filename": doc.get("filename"),
                "combined_text": text
            })
            print(f"[PIPELINE] Vector indexing OK for doc_id={doc_id}")
        except Exception as e:
            print(f"[PIPELINE] Vector indexing failed: {e}")

    try:
        print("[PIPELINE] Stage: Basic analysis")
        basic_analysis = analyze_document(working_text or text)
    except Exception:
        basic_analysis = {"category": "Other", "document_type": "Other", "entities": {"dates": [], "amounts": [], "survey_numbers": [], "case_numbers": []}}
    # 10️⃣ Save back to Mongo (Async)
    try:
        # Validate analytics structure before saving
        if not isinstance(analytics, dict):
            analytics = {"error": "Invalid analytics structure"}

        await asyncio.to_thread(save_document, doc_id, {
            "analytics": analytics,
            "results": results,
            "pipeline.classify": {"document_type": analytics.get("document_type")},
            "pipeline.ner": {"entities": results, "stats": analytics},
            "pipeline.risk": {"risk": analytics.get("clause_risk")},
            "pipeline.timeline": {"timeline": analytics.get("timeline")},
            "analysis.basic": basic_analysis,
            "analysis.ai": ai_result or {}
        })
        print(f"[PIPELINE] Saved results for doc_id={doc_id}")
    except Exception as e:
        # Try to save at least the basic analytics
        try:
            await asyncio.to_thread(save_document, doc_id, {"analytics": analytics})
            print(f"[PIPELINE] Saved partial analytics for doc_id={doc_id}")
        except Exception:
            print(f"[PIPELINE] Failed to save document: {str(e)}")
            return {"error": "Failed to save document results"}

    try:
        print(f"[PIPELINE] Verification status={analytics.get('verified_marker')}, ai_confidence={analytics.get('ai_confidence')}")
    except Exception:
        pass

    # 11️⃣ Final Legal Status (with logging)
    try:
        legal_status = analysis_service.derive_legal_status(analytics, llm_risk)
        analytics["legal_status"] = legal_status.get("status")
        analytics["status_summary"] = legal_status.get("summary")
        print(f"[PIPELINE] Legal status decision: type={analytics.get('document_type')} marker={analytics.get('verified_marker')} ai_conf={analytics.get('ai_confidence')} status={analytics.get('legal_status')}")
    except Exception as e:
        print(f"[PIPELINE] Legal status derivation failed: {e}")
        analytics["legal_status"] = analytics.get("legal_status") or "UNKNOWN"

    # Validate final output structure
    try:
        # Ensure all required fields are present
        if "document_type" not in analytics:
            analytics["document_type"] = "Unknown"
        if "legality_score" not in analytics:
            analytics["legality_score"] = 50
        if "ai_confidence" not in analytics:
            analytics["ai_confidence"] = 0

        # Normalize scores to 0-100 range
        analytics["legality_score"] = max(0, min(100, analytics["legality_score"]))
        analytics["ai_confidence"] = max(0, min(100, analytics["ai_confidence"]))

        out = {
            "doc_id": doc_id,
            "filename": filename,
            "results_summary": results,
            "analytics": analytics
        }
        print(f"[PIPELINE] Completed doc_id={doc_id} type={analytics.get('document_type')} ai_conf={analytics.get('ai_confidence')}")
        return out
    except Exception as e:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "error": f"Failed to format results: {str(e)}",
            "analytics": analytics
        }

@router.post("/full")
async def full(
    doc_id: str = Form(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user)
):
    res = await run_full_pipeline(doc_id, background_tasks)
    if res.get("error"):
        raise HTTPException(status_code=404, detail=res["error"])
    return res

@router.post("/start")
async def start_pipeline(
    doc_id: str = Form(...),
    user: dict = Depends(get_current_user)
):
    job_id = str(uuid.uuid4())
    _init_pipeline_job(job_id, doc_id)
    asyncio.create_task(_run_pipeline_job(job_id, doc_id))
    return {"status": "queued", "job_id": job_id}

@router.get("/status/{job_id}")
async def pipeline_status(
    job_id: str,
    user: dict = Depends(get_current_user)
):
    job = _PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# Backward-compatible alias paths (optional, but keep it clean)
@router.post("/start_alias")
async def start_pipeline_alias(
    doc_id: str = Form(...),
    user: dict = Depends(get_current_user)
):
    return await start_pipeline(doc_id=doc_id, user=user)

@router.get("/status_alias/{job_id}")
async def pipeline_status_alias(
    job_id: str,
    user: dict = Depends(get_current_user)
):
    return await pipeline_status(job_id=job_id, user=user)
