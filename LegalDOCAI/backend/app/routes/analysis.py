from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from backend.app.database import documents_collection as collection, db
from backend.app.core.deps import get_openai_client, get_async_openai_client
from backend.app.core.config import OPENAI_MODEL
from backend.app.services import analysis_service
from backend.app.services.risk_service import compute_combined_risk_index
from backend.app.routes.pipeline import run_full_pipeline
from typing import List
import asyncio
import time
import json
import datetime

router = APIRouter(tags=["analysis"])


class HybridClassifyResponse(BaseModel):
    document_name: str
    document_type: str
    legal_status: str
    confidence_score: str
    risk_level: str
    missing_clauses: List[str]
    reasoning_summary: str


class DeterministicClassifyResponse(BaseModel):
    document_type: str
    rule_match_score: str
    ml_confidence: str
    final_confidence: str
    classification_source: str
    stability_hash: str

def _local_explain(corpus: str, doc_type: str = None) -> str:
    label = doc_type or analysis_service.classify_document_type(corpus)
    label = label or "Unknown Document"
    t = label.lower()

    ownership = "Record only."
    if label in {"Sale Deed", "Gift Deed", "Partition Deed", "Release Deed", "Conveyance Deed"}:
        ownership = "Usually strong proof of ownership if properly registered."
    elif "rental agreement" in t or "lease" in t:
        ownership = "Does not prove ownership. It only shows the right to use or occupy the property."
    elif "patta" in t or "encumbrance" in t or "survey land register" in t or "tslr" in t:
        ownership = "Record only. It shows land records but is not final proof of ownership."
    elif "certificate" in t or "card" in t or "id" in t or "licence" in t or "license" in t or "statement" in t or "passbook" in t:
        ownership = "Record only. It usually proves identity, registration, or status—not ownership."
    else:
        ownership = "Record only unless a registered title deed is provided."

    summary = ""
    try:
        summary = analysis_service.humanize_text_ai(corpus)
    except Exception:
        summary = ""

    parts = [
        f"This document appears to be: {label}.",
        "Legal importance: It is an official document that records or supports a legal fact.",
        f"Ownership: {ownership}",
    ]
    if summary:
        parts.append(f"Simple summary: {summary}")
    return "\n".join(parts)

@router.post("/advanced")
async def analysis_advanced(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    filename = None
    if not corpus and doc_id:
        d = await asyncio.to_thread(collection.find_one, {"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "filename": 1, "analytics": 1})
        corpus = d.get("combined_text","") if d else ""
        filename = d.get("filename") if d else None
        existing_type = (d.get("analytics") or {}).get("document_type") if d else None
        if corpus.strip():
            # Reuse central pipeline for consistency with upload/modules/document routes.
            full = await run_full_pipeline(doc_id)
            if full.get("error"):
                raise HTTPException(status_code=400, detail=full.get("error"))
            analytics = full.get("analytics") or {}
            total = {
                "results_summary": full.get("results_summary") or {},
                "analytics": analytics,
                "filename": full.get("filename") or filename,
                "clause_risk": analytics.get("clause_risk") or {},
                "jurisdiction_ai": analytics.get("jurisdiction_checks") or {},
                "jurisdiction_curated": analytics.get("jurisdiction_curated") or {},
                "signer_validation": analytics.get("signer_validation") or {},
                "registration_analysis": analytics.get("registration_analysis") or {},
                "verification_ai": {
                    "marker": analytics.get("verified_marker"),
                    "ai_confidence": analytics.get("ai_confidence"),
                },
                "mandatory_fields": analytics.get("mandatory_fields") or {},
                "key_facts": {},
                "humanized_text": analytics.get("summary_simple") or analytics.get("summary") or "",
            }
            return {"pipeline": [], "summary": total}
    if not corpus:
        # Instead of 400 error, return empty valid structure to prevent frontend hanging
        return {
            "pipeline": [], 
            "summary": {
                "analytics": {"document_type": "Unknown", "legality_score": 0, "filename": filename}, 
                "key_facts": {}, 
                "clause_risk": {"clauses": []}
            }
        }
        
    # Parallel execution using asyncio.gather for speed
    
    # 1. Base NLP & Stats (This must be await-ed first as it produces working_text)
    rs, an, working_text = await analysis_service.analyze_text_overall_async(corpus)
    
    # 2. Key Facts, Clauses, Verification, etc. (Parallel)
    tasks = {
        "key_facts": analysis_service.extract_key_facts_llm_async(corpus),
        "clause": analysis_service.clause_risk_detection_async(corpus),
        "verification": analysis_service.ask_openai_for_verification_and_confidence_async(corpus),
        "juris_ai": analysis_service.jurisdiction_specific_checks_async(corpus),
        "juris_cur": analysis_service.curated_jurisdiction_templates_async(corpus),
        "signers": analysis_service.validate_signers_async(corpus, rs.get("signers", [])),
        "reg": analysis_service.analyze_registration_async(corpus),
        "mand": analysis_service.check_mandatory_fields_async(corpus),
        "humanized": analysis_service.humanize_text_ai_async(corpus)
    }
    if not locals().get("existing_type"):
        tasks["classification"] = analysis_service.classify_document_type_async(corpus)
    
    # Run all analysis tasks in parallel
    task_keys = list(tasks.keys())
    results_list = await asyncio.gather(*[tasks[k] for k in task_keys], return_exceptions=True)
    
    # Map results back to variables
    results_map = {}
    for i, key in enumerate(task_keys):
        val = results_list[i]
        if isinstance(val, Exception):
            print(f"[ERROR] analysis_advanced: Task {key} failed: {val}")
            # Fallbacks
            if key == "clause": results_map[key] = {"clauses": [], "overall_risk_score": 0}
            elif key == "verification": results_map[key] = {"marker": "UNVERIFIED", "ai_confidence": 40}
            elif key == "signers": results_map[key] = {"signers": [], "overall_signer_score": 20}
            elif key == "juris_cur": results_map[key] = {"jurisdiction": None, "checks": [], "overall_compliance_score": 40}
            else: results_map[key] = {}
        else:
            results_map[key] = val

    # Ensure baseline scores if missing from successful tasks
    if not results_map.get("clause", {}).get("overall_risk_score"):
        if "clause" not in results_map: results_map["clause"] = {}
        results_map["clause"]["overall_risk_score"] = 0
    
    if not results_map.get("signers", {}).get("overall_signer_score"):
        if "signers" not in results_map: results_map["signers"] = {}
        results_map["signers"]["overall_signer_score"] = 20
        
    if not results_map.get("juris_cur", {}).get("overall_compliance_score"):
        if "juris_cur" not in results_map: results_map["juris_cur"] = {}
        results_map["juris_cur"]["overall_compliance_score"] = 40

    key_facts = results_map["key_facts"]
    clause = results_map["clause"]
    v = results_map["verification"]
    juris_ai = results_map["juris_ai"]
    juris_cur = results_map["juris_cur"]
    signers = results_map["signers"]
    reg = results_map["reg"]
    mand = results_map["mand"]
    humanized = results_map["humanized"]

    # Normalize verification for legacy keys
    if isinstance(v, dict):
        if v.get("ai_confidence") is None and v.get("confidence") is not None:
            v["ai_confidence"] = v.get("confidence")
        if not v.get("marker") and v.get("status"):
            v["marker"] = v.get("status")

    # Construct Response
    total = {
        "results_summary": rs,
        "analytics": {**an, "document_type": (locals().get("existing_type") or results_map.get("classification") or "Generic Legal Document")},
        "filename": filename,
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
    total["analytics"].update({
        "clause_risk": clause,
        "jurisdiction_ai": juris_ai,
        "jurisdiction_curated": juris_cur,
        "signer_validation": signers,
        "registration_analysis": reg,
        "mandatory_fields": mand,
        "verified_marker": v.get("marker"),
        "ai_confidence": v.get("ai_confidence"),
        "filename": filename
    })
    
    total["analytics"]["legality_score"] = analysis_service.compute_legality_score(total["analytics"])
    total["analytics"]["combined_risk_index"] = compute_combined_risk_index(total["analytics"])
    total["analytics"]["ai_detection_score"] = analysis_service.compute_ai_detection_score(working_text or corpus)
    
    return {"pipeline": [], "summary": total}

@router.post("/explain")
async def analysis_explain(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    doc_type = None
    if not corpus and doc_id:
        d = await asyncio.to_thread(collection.find_one, {"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "analytics": 1})
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
        client = get_async_openai_client()
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        explanation = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        explanation = ""
    if not explanation:
        explanation = await asyncio.to_thread(_local_explain, corpus, doc_type)
    if doc_id and explanation:
        try:
            await asyncio.to_thread(collection.update_one, {"doc_id": doc_id}, {"$set": {"analytics.explanation_simple": explanation}})
        except Exception:
            pass
    return {"explanation": explanation, "document_type": doc_type or await analysis_service.classify_document_type_async(corpus)}

@router.post("/summary/translate")
async def translate_summary(doc_id: str = Form(...), target_lang: str = Form("English")):
    d = await asyncio.to_thread(collection.find_one, {"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "analytics": 1})
    if not d:
        raise HTTPException(status_code=404, detail="Document not found")
    an = d.get("analytics") or {}
    base_text = d.get("combined_text", "") or ""
    summary = an.get("summary_simple") or an.get("summary") or ""
    if not summary.strip():
        summary = analysis_service.summarize_fallback(base_text)
    translated = await analysis_service.translate_text_simple_async(summary, target_lang)
    try:
        await asyncio.to_thread(collection.update_one, {"doc_id": doc_id}, {"$set": {"analytics.summary_translated": {"lang": target_lang, "text": translated}}})
    except Exception:
        pass
    return {"doc_id": doc_id, "summary_translated": {"lang": target_lang, "text": translated}}

@router.post("/hybrid/classify", response_model=HybridClassifyResponse)
async def hybrid_classify(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    filename = ""
    if not corpus and doc_id:
        d = await asyncio.to_thread(collection.find_one, {"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "filename": 1})
        if d:
            corpus = d.get("combined_text", "") or ""
            filename = d.get("filename") or ""
    if not corpus:
        return {
            "document_name": filename or "",
            "document_type": "Unknown",
            "legal_status": "UNKNOWN",
            "confidence_score": "0%",
            "risk_level": "LOW",
            "missing_clauses": [],
            "reasoning_summary": "No text provided."
        }
    out = analysis_service.hybrid_classify_verify(corpus, filename or "")
    return out


@router.post("/deterministic/classify", response_model=DeterministicClassifyResponse)
async def deterministic_classify(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = await asyncio.to_thread(
            collection.find_one,
            {"doc_id": doc_id},
            {"_id": 0, "combined_text": 1, "text": 1},
        )
        if d:
            corpus = (d.get("combined_text") or d.get("text") or "")
    return analysis_service.deterministic_classify_document(corpus)

@router.post("/risk")
async def analysis_risk(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = await asyncio.to_thread(collection.find_one, {"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text","") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    result = await analysis_service.llm_risk_validation_async(corpus)
    if doc_id:
        try:
            await asyncio.to_thread(collection.update_one, {"doc_id": doc_id}, {"$set": {"analytics.llm_risk": result}})
        except Exception:
            pass
    return {"risk": result}


@router.post("/structural/audit")
async def structural_audit(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    classified_type = None
    if not corpus and doc_id:
        d = await asyncio.to_thread(
            collection.find_one,
            {"doc_id": doc_id},
            {"_id": 0, "combined_text": 1, "text": 1, "analytics.document_type": 1},
        )
        if d:
            corpus = (d.get("combined_text") or d.get("text") or "")
            classified_type = ((d.get("analytics") or {}).get("document_type") if isinstance(d.get("analytics"), dict) else None)
    if not corpus:
        return {
            "document_type": "NOT A LEGAL DOCUMENT",
            "sections_missing": [],
            "note": "No legal structural template applicable."
        }

    result = analysis_service.structural_audit_strict(corpus, classified_type)
    if doc_id:
        try:
            await asyncio.to_thread(
                collection.update_one,
                {"doc_id": doc_id},
                {"$set": {"analytics.structural_audit": result}},
            )
        except Exception:
            pass
    return result

@router.post("/feedback")
async def analysis_feedback(doc_id: str = Form(None), tag: str = Form(None), data: str = Form(None)):
    import time as _time
    payload = {
        "doc_id": doc_id,
        "tag": tag,
        "data": json.loads(data) if data else None,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        if db is not None:
            fb = db["training_feedback"]
            await asyncio.to_thread(fb.insert_one, payload)
        else:
            from backend.app.core.config import UPLOAD_FOLDER
            import os
            dir_path = os.path.join(UPLOAD_FOLDER, "trust_safety")
            os.makedirs(dir_path, exist_ok=True)
            ts_ms = int(_time.time() * 1000)
            fname = os.path.join(dir_path, f"analysis_feedback_{ts_ms}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")
    return {"status": "ok"}
