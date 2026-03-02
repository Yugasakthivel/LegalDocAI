from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from typing import List, Dict, Any

from backend.app.services import ocr_service, analysis_service
from backend.app.services.risk_service import clause_risk_detection as risk_llm
from backend.app.services.compliance_service import jurisdiction_checks, curated_checks, mandatory_fields
from backend.app.services.verification_service import verify_document, validate_signers
from backend.app.services.storage_service import create_initial_document, get_document, save_document
from backend.app.vectorstore import add_document, fetch_document_by_id, search_doc_first
from backend.app.core.config import UPLOAD_FOLDER
from backend.app.core.deps import get_current_user, is_openai_ready
from backend.app.routes.pipeline import run_full_pipeline
from backend.app.services.offline_rag_service import get_rag_mode, generate_answer_offline
import backend.app.vectorstore as vectorstore
from backend.app.services.indian_legal_verifier import IndianLegalVerifier
import asyncio

try:
    from backend.app.services.trust_safety_service import PIIDetectionService, AccountabilityService
    _pii = PIIDetectionService()
    _acct = AccountabilityService()
except Exception:
    _pii = None
    _acct = None
# Optional deps for Module 1 forensic checks
try:
    import cv2
except Exception:
    cv2 = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

from PIL import Image, ImageFilter
import io
import numpy as np

router = APIRouter(prefix="/modules", tags=["modules"])


def _openai_configured() -> bool:
    return is_openai_ready()

# Helper to persist incremental pipeline status

def _update_pipeline(doc_id: str, field: str, value: Any):
    try:
        save_document(doc_id, {f"pipeline.{field}": value})
    except Exception:
        pass


# MODULE 1 – Document Ingestion
@router.post("/ingest")
async def module_ingest(file: UploadFile = File(...), ocr_lang: str = Form("auto"), ocr_engine: str = Form(None), user: dict = Depends(get_current_user)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate extension and size
    allowed = {"pdf","png","jpg","jpeg","docx","xls","xlsx","txt"}
    ext = (file.filename.split(".")[-1] or "").lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: pdf, png, jpg, jpeg, docx, xls, xlsx, txt."
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    processed_dir = os.path.join(UPLOAD_FOLDER, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Save temp
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract metadata (PDF/Image)
    metadata: Dict[str, Any] = {"filename": file.filename}
    try:
        size_bytes = os.path.getsize(temp_path)
        metadata["size_bytes"] = size_bytes
    except Exception:
        pass

    if ext == "pdf" and PdfReader:
        try:
            reader = PdfReader(temp_path)
            metadata["pdf_pages"] = len(reader.pages)
            if reader.metadata:
                metadata["pdf_meta"] = {k: str(v) for k, v in dict(reader.metadata).items()}
        except Exception:
            pass
    elif ext in {"png","jpg","jpeg"}:
        try:
            with Image.open(temp_path) as im:
                metadata["image_mode"] = im.mode
                metadata["image_size"] = im.size
                metadata["dpi"] = im.info.get("dpi")
        except Exception:
            pass
    elif ext == "docx":
        metadata["docx"] = True
    elif ext in {"xls","xlsx"}:
        metadata["excel"] = True

    # Image enhancement + preview saving (for images or first page raster)
    clean_file = None
    try:
        if ext in {"png","jpg","jpeg"}:
            with Image.open(temp_path) as im:
                im2 = im.convert("L").filter(ImageFilter.SHARPEN)
                if cv2:
                    # additional denoise/threshold via OpenCV if available
                    pass
                out_path = os.path.join(processed_dir, f"clean_{uuid.uuid4().hex}.png")
                im2.save(out_path)
                clean_file = out_path
        elif ext == "pdf":
            # keep original PDF as clean reference; preview could be generated upstream if needed
            clean_file = temp_path
    except Exception:
        clean_file = temp_path

    # Simple forgery heuristics (placeholders): blur + metadata anomalies
    forgery_flags: List[str] = []
    try:
        if cv2 and ext in {"png","jpg","jpeg"}:
            data = np.frombuffer(open(temp_path,'rb').read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                var = cv2.Laplacian(img, cv2.CV_64F).var()
                if var < 15: forgery_flags.append("Very blurry scan")
    except Exception:
        pass
    if metadata.get("pdf_meta"):
        if any(k for k in metadata["pdf_meta"].keys() if "Producer" in k or "Creator" in k) is False:
            forgery_flags.append("Missing standard PDF producer metadata")

    # OCR language selection
    file_type = ocr_service.detect_file_type(file.filename)
    resolved_lang = ocr_service.resolve_ocr_lang_for_file(temp_path, file_type, ocr_lang)

    # Extract text now so the rest of pipeline can work off DB state
    pages: List[str] = ocr_service.extract_text_by_filetype(temp_path, file_type, ocr_lang=resolved_lang, ocr_engine=ocr_engine)
    combined_text = "\n".join(pages)

    # Authenticity score: combine length density + heuristics penalty
    text_len = len(combined_text.strip())
    page_count = max(1, len(pages))
    density = min(1.0, text_len / (page_count * 5000))
    authenticity_score = int(round(70 + 30 * density))
    if forgery_flags:
        authenticity_score = max(0, authenticity_score - min(30, 10 * len(forgery_flags)))

    doc_id = str(uuid.uuid4())
    create_initial_document(doc_id, file.filename, file_type, combined_text, pages=pages)

    payload = {
        "doc_id": doc_id,
        "clean_file": clean_file,
        "auth_score": authenticity_score,
        "metadata": metadata,
        "ocr_lang": resolved_lang,
        "pages": page_count,
    }
    _update_pipeline(doc_id, "ingest", payload)
    try:
        save_document(doc_id, {
            "file_info": {
                "filename": file.filename,
                "type": file_type,
                "pages": page_count,
                "ocr_lang": resolved_lang
            },
            "analytics": {
                "pages": page_count
            }
        })
        doc_type = analysis_service.classify_document_type(combined_text[:6000])
        doc_category = analysis_service.classify_document_category(combined_text[:6000])
        two_step = analysis_service.classify_legal_two_step(combined_text[:6000], doc_type)
        save_document(doc_id, {
            "analytics.document_type": doc_type,
            "analytics.document_category": doc_category,
            "analytics.is_legal_document": bool(two_step.get("is_legal")),
            "analytics.is_indian_legal_document": bool(two_step.get("is_indian"))
        })
        payload["document_type"] = doc_type
        payload["document_category"] = doc_category
        payload["is_legal_document"] = bool(two_step.get("is_legal"))
        payload["is_indian_legal_document"] = bool(two_step.get("is_indian"))
    except Exception:
        pass
    try:
        if _pii:
            pii = _pii.audit_compliance(combined_text[:10000])
            save_document(doc_id, {"analytics.pii_summary": pii})
            payload["pii_summary"] = pii
        if _acct:
            _acct.log_event(doc_id, "M1_Ingest", "pii_scan", decision="completed", details={"size_bytes": metadata.get("size_bytes")})
    except Exception:
        pass
    # Do not delete temp_path immediately if used as clean_file
    try:
        if clean_file != temp_path:
            os.remove(temp_path)
    except Exception:
        pass

    return JSONResponse(payload)


# MODULE 2 – OCR & Layout (uses existing OCR engine outputs)
@router.post("/ocr")
async def module_ocr(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    if not text:
        raise HTTPException(status_code=400, detail="Document has no extracted text")

    # Enhanced layout grouping
    headings = analysis_service.detect_clause_headings(text)
    structured_layout = analysis_service.extract_structured_layout(text)
    
    out = {
        "doc_id": doc_id,
        "extracted_text": text,
        "layout": {
            "headings": headings,
            "structured_layout": structured_layout,
            "page_count": len(doc.get("pages") or [])
        },
    }
    _update_pipeline(doc_id, "ocr", {"pages": out["layout"]["page_count"], "layout": out["layout"]})
    save_document(doc_id, {"analytics.layout": out["layout"]})
    return JSONResponse(out)


# MODULE 3 – Language Detection & Translation
@router.post("/lang")
async def module_language(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    lang = analysis_service.detect_language(text[:4000])

    normalized_text = text
    translation_triggered = False
    if lang and lang != "English":
        try:
            normalized_text = await analysis_service.translate_to_english_async(text[:6000], lang)
            translation_triggered = True
        except Exception as e:
            print(f"[DEBUG] module_language translation failed: {e}")

    out = {
        "doc_id": doc_id, 
        "language": lang, 
        "normalized_text": normalized_text,
        "translation_triggered": translation_triggered
    }
    _update_pipeline(doc_id, "lang", out)
    save_document(doc_id, {"analytics.detected_language": lang})
    return JSONResponse(out)


# MODULE 4 – Document Classification
@router.post("/classify")
async def module_classify(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    doc_type = analysis_service.classify_document_type(text)
    out = {"doc_id": doc_id, "document_type": doc_type, "confidence": None}
    _update_pipeline(doc_id, "classify", out)
    save_document(doc_id, {"analytics.document_type": doc_type})
    try:
        if _acct:
            _acct.log_event(doc_id, "M4_Classify", "classification", decision=doc_type, details={})
    except Exception:
        pass
    return JSONResponse(out)


# MODULE 5 – Legal NER
@router.post("/ner")
async def module_ner(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    results, analytics, working_text = await analysis_service.analyze_text_overall_async(text)
    extra = analysis_service.extract_indian_entities(working_text or text)
    entities = results.get("entities", {}) if isinstance(results, dict) else {}
    contact_info = results.get("contact_info", {}) if isinstance(results, dict) else {}
    phones_raw = contact_info.get("phones", []) if isinstance(contact_info, dict) else []
    phones = []
    for p in (phones_raw or []):
        if isinstance(p, dict):
            val = p.get("normalized") or p.get("number")
            if val:
                phones.append(str(val))
        elif p:
            phones.append(str(p))
    phones = list(dict.fromkeys(phones))
    out = {
        "doc_id": doc_id,
        "entities": {
            "names": entities.get("names", []),
            "organizations": entities.get("organizations", []),
            "emails": contact_info.get("emails", []),
            "phones": phones,
            "dates": results.get("dates", []),
            "signers": results.get("signers", []),
            "parties": extra.get("parties", []),
            "role_parties": extra.get("role_parties", []),
            "money": extra.get("money", []),
            "survey_numbers": extra.get("survey_numbers", []),
            "registration_numbers": extra.get("registration_numbers", []),
            "case_numbers": extra.get("case_numbers", []),
            "courts": extra.get("courts", []),
            "land_types": extra.get("land_types", [])
        },
        "stats": analytics,
        "property_details": extra.get("property_details", "")
    }
    _update_pipeline(doc_id, "ner", out)
    save_document(doc_id, {"results": results, "analytics": analytics})
    return JSONResponse(out)


# MODULE 6 – Clause Detection
@router.post("/clauses")
async def module_clauses(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    clause = analysis_service.clause_risk_detection(text)
    doc_type = (doc.get("analytics", {}) or {}).get("document_type") or analysis_service.classify_document_type(text)
    checklist = mandatory_fields(text, doc_type)
    out = {"doc_id": doc_id, "clause_risk": clause, "checklist": checklist}
    _update_pipeline(doc_id, "clauses", out)
    save_document(doc_id, {"analytics.clause_risk": clause, "analytics.mandatory_fields": checklist})
    return JSONResponse(out)


# MODULE 7 – Risk Detection
@router.post("/risk")
async def module_risk(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    clause = risk_llm(text)
    juris_ai = jurisdiction_checks(text)
    juris_cur = curated_checks(text)
    out = {"doc_id": doc_id, "risk": clause, "jurisdiction_ai": juris_ai, "jurisdiction_curated": juris_cur}
    _update_pipeline(doc_id, "risk", out)
    save_document(doc_id, {"analytics.clause_risk": clause, "analytics.jurisdiction_checks": juris_ai, "analytics.jurisdiction_curated": juris_cur})
    try:
        if _acct:
            overall = (clause or {}).get("overall_risk")
            _acct.log_event(doc_id, "M7_Risk", "risk_detection", decision=overall, details={"risk_score": (clause or {}).get("overall_risk_score")})
    except Exception:
        pass
    return JSONResponse(out)


# MODULE 8 – Explainable AI
@router.post("/explain")
async def module_explain(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    explanation = analysis_service.legal_explanation_engine(text)
    out = {"doc_id": doc_id, "explanation": explanation}
    _update_pipeline(doc_id, "explain", out)
    save_document(doc_id, {"analytics.explanation": explanation})
    return JSONResponse(out)


# MODULE 9 – Timeline Extraction
@router.post("/timeline")
async def module_timeline(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    tl = analysis_service.build_timeline(text)
    out = {"doc_id": doc_id, "timeline": tl}
    _update_pipeline(doc_id, "timeline", out)
    save_document(doc_id, {"analytics.timeline": tl})
    return JSONResponse(out)


# MODULE 10 – Chatbot (RAG)
@router.post("/rag/query")
async def module_rag_query(doc_id: str = Form(...), question: str = Form(...), user: dict = Depends(get_current_user)):
    if not question:
        raise HTTPException(status_code=400, detail="Question required")
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    mode_cfg = get_rag_mode()
    openai_ready = _openai_configured()
    text = (doc.get("combined_text") or doc.get("text") or "")
    if (mode_cfg == "local") or (not openai_ready):
        a = doc.get("analytics") or {}
        a = {**a, "filename": doc.get("filename")}
        data = generate_answer_offline(question, text, a)
        return JSONResponse({
            "doc_id": doc_id,
            "answer": data.get("answer") or "",
            "source": data.get("source") or "offline_rag",
            "chunks_used": data.get("chunks_used") or 0,
            "confidence": data.get("confidence"),
            "mode": "offline",
        })
    if doc and (doc.get("combined_text") or doc.get("text")):
        try:
            add_document({"doc_id": doc_id, "filename": doc.get("filename"), "combined_text": doc.get("combined_text") or doc.get("text")})
        except Exception:
            pass
    sources = search_doc_first(question, doc_id, top_k=3) or []
    context = (doc.get("combined_text") or doc.get("text") or "")[:15000]
    try:
        answer = await asyncio.to_thread(analysis_service.answer_with_fallback, question, context)
        return JSONResponse({
            "doc_id": doc_id,
            "answer": answer or "",
            "source": "openai_rag",
            "chunks_used": len(sources),
            "confidence": None,
            "mode": "openai",
        })
    except Exception:
        data = generate_answer_offline(question, text, {})
        return JSONResponse({
            "doc_id": doc_id,
            "answer": data.get("answer") or "",
            "source": data.get("source") or "offline_rag",
            "chunks_used": data.get("chunks_used") or 0,
            "confidence": data.get("confidence"),
            "mode": "offline",
        })

@router.get("/rag/status")
async def rag_status(user: dict = Depends(get_current_user)):
    mode_cfg = get_rag_mode()
    openai_ready = _openai_configured()
    local_ready = bool(getattr(vectorstore, "_fe_pipeline", None))
    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL") or getattr(vectorstore, "_fe_model_name", "")
    if mode_cfg == "local":
        active = "offline"
    elif openai_ready and mode_cfg in {"openai", "auto"}:
        active = "openai"
    else:
        active = "offline"
    return JSONResponse({
        "active_mode": active,
        "configured_mode": mode_cfg,
        "openai_ready": bool(openai_ready),
        "local_embed_ready": bool(local_ready),
        "model_name": model_name,
    })


# MODULE 11 – Document Comparison (file or text)
@router.post("/compare")
async def module_compare(
    doc_id: str = Form(...),
    other_text: str = Form(None),
    other_file: UploadFile = File(None),
    ocr_lang: str = Form(None),
    ocr_engine: str = Form(None),
    user: dict = Depends(get_current_user)
):
    import difflib
    from backend.app.core.deps import get_openai_client
    from backend.app.core.config import OPENAI_MODEL

    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    base = doc.get("combined_text") or ""

    other = other_text or ""
    if not other and other_file:
        tmpdir = os.path.join(UPLOAD_FOLDER, "compare_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        temp_path = os.path.join(tmpdir, f"cmp_{uuid.uuid4().hex}_{other_file.filename}")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(other_file.file, f)
        ext = ocr_service.detect_file_type(other_file.filename)
        eff_lang = ocr_service.resolve_ocr_lang_for_file(temp_path, ext, ocr_lang or "auto")
        pages = ocr_service.extract_text_by_filetype(temp_path, ext, ocr_lang=eff_lang, ocr_engine=ocr_engine)
        other = "\n".join(pages or [])
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if not other:
        return JSONResponse({"doc_id": doc_id, "diff": {"removed": [], "added": []}, "note": "Provide other_text or other_file"})

    # 1. Structural Diff
    d = difflib.unified_diff(base.splitlines(), other.splitlines(), lineterm="")
    diff_lines = list(d)
    added = [l[1:].strip() for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in diff_lines if l.startswith("-") and not l.startswith("---")]

    # 2. Semantic Analysis (if LLM is ready)
    semantic_report = {}
    if is_openai_ready():
        try:
            prompt = (
                "Compare these two legal document excerpts.\n"
                "Focus on critical differences: Parties, Dates, Payment, Property Details.\n"
                "Return JSON with keys: 'critical_differences' (list), 'summary' (string).\n\n"
                "DOC 1:\n" + base[:2000] + "\n\nDOC 2:\n" + other[:2000]
            )
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            raw = resp.choices[0].message.content or ""
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                semantic_report = json.loads(raw[start:end])
        except Exception as e:
            print(f"[DEBUG] module_compare semantic analysis failed: {e}")

    out = {
        "doc_id": doc_id,
        "structural_diff": {
            "added_count": len(added),
            "removed_count": len(removed),
            "added_samples": added[:100],
            "removed_samples": removed[:100],
        },
        "semantic_report": semantic_report
    }
    _update_pipeline(doc_id, "compare", out)
    return JSONResponse(out)


# MODULE 12 – Report Generation (JSON placeholder)
@router.api_route("/report", methods=["GET", "POST"])
async def module_report(
    doc_id: str = None,
    doc_id_form: str = Form(None)
):
    actual_doc_id = doc_id or doc_id_form
    if not actual_doc_id:
        raise HTTPException(status_code=400, detail="doc_id required")
    
    doc = fetch_document_by_id(actual_doc_id) or get_document(actual_doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    analytics = doc.get("analytics") or {}
    results = doc.get("results") or {}
    report = {
        "doc_id": actual_doc_id,
        "filename": doc.get("filename"),
        "document_type": analytics.get("document_type"),
        "overall_risk": (analytics.get("clause_risk") or {}).get("overall_risk"),
        "risk_score": (analytics.get("clause_risk") or {}).get("overall_risk_score"),
        "metrics": {
            "total_risks": len((analytics.get("clause_risk") or {}).get("clauses", [])),
            "missing_clauses": len((analytics.get("mandatory_fields") or {}).get("missing", [])),
            "pages": len(doc.get("pages") or []),
            "confidence": analytics.get("ai_confidence"),
        },
        "entities": results,
        "timeline": analytics.get("timeline", []),
    }
    try:
        vr = analytics.get("verification_report")
        if not vr:
            _verifier = IndianLegalVerifier()
            text = (doc.get("combined_text") or doc.get("text") or "")
            try:
                vr = _verifier.verify(text)
            except Exception as e:
                vr = {"error": str(e)}
        report["verification_report"] = vr
    except Exception as e:
        report["verification_report"] = {"error": str(e)}
    return JSONResponse(report)

def _legal_doc_confidence(text: str, stats: Dict[str, Any], doc_type: str = None) -> int:
    try:
        t = (text or "").lower()
        keys = [
            "agreement","clause","party","consideration","termination","confidential",
            "arbitration","jurisdiction","indemnity","warranty","liability","governing law"
        ]
        kscore = sum(1 for k in keys if k in t) / max(1, len(keys))
        ents = float(stats.get("total_names",0) + stats.get("total_emails",0) + stats.get("total_phones",0) + stats.get("total_clauses",0))
        escore = min(1.0, ents/30.0)
        dscore = 0.2 if doc_type and doc_type not in ("Unknown","Other Legal Document") else 0.0
        conf = int(round(100 * min(1.0, 0.55*kscore + 0.35*escore + 0.10 + dscore)))
        return max(0, min(100, conf))
    except Exception:
        return 50

# One-shot pipeline orchestrator for the new method
@router.post("/pipeline/new_full")
async def pipeline_new_full(doc_id: str = Form(...), user: dict = Depends(get_current_user)):
    # Use the central full pipeline so all routes produce identical outputs/shapes.
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not (doc.get("combined_text") or "").strip():
        raise HTTPException(status_code=400, detail="No text available for analysis")

    result = await run_full_pipeline(doc_id)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return JSONResponse(result)
