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

    # Basic layout grouping via detected clause headings
    headings = analysis_service.detect_clause_headings(text)
    out = {
        "doc_id": doc_id,
        "extracted_text": text,
        "layout": {"headings": headings},
    }
    _update_pipeline(doc_id, "ocr", {"pages": len(doc.get("pages") or []), "layout": out["layout"]})
    return JSONResponse(out)


# MODULE 3 – Language Detection & Translation
@router.post("/lang")
async def module_language(doc_id: str = Form(...)):
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = doc.get("combined_text") or ""
    lang = ocr_service.detect_language_simple(text)

    # Placeholder translation: for now, pass through text. Hook MarianMT or API later.
    normalized_text = text

    out = {"doc_id": doc_id, "language": lang, "normalized_text": normalized_text}
    _update_pipeline(doc_id, "lang", out)
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
    out = {
        "doc_id": doc_id,
        "entities": {
            "names": entities.get("names", []),
            "organizations": entities.get("organizations", []),
            "emails": contact_info.get("emails", []),
            "phones": contact_info.get("phones", []),
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
    checklist = mandatory_fields(text)
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
    if not _openai_configured():
        return JSONResponse({"error": "OpenAI not configured", "code": "OPENAI_NOT_CONFIGURED"}, status_code=503)
    # Ensure document is indexed
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if doc and (doc.get("combined_text") or doc.get("text")):
        try:
            add_document({"doc_id": doc_id, "filename": doc.get("filename"), "combined_text": doc.get("combined_text") or doc.get("text")})
        except Exception:
            pass
    sources = search_doc_first(question, doc_id, top_k=3) or []
    return JSONResponse({"doc_id": doc_id, "answers": sources})


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

    base_sents = [s.strip() for s in base.split("\n") if s.strip()]
    other_sents = [s.strip() for s in other.split("\n") if s.strip()]
    removed = [s for s in base_sents if s not in other_sents]
    added = [s for s in other_sents if s not in base_sents]
    return JSONResponse({"doc_id": doc_id, "diff": {"removed": removed[:200], "added": added[:200]}})


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
    # This assumes ingest already saved combined_text. If not, we just operate on existing doc.
    doc = fetch_document_by_id(doc_id) or get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    text = doc.get("combined_text") or ""
    if not text:
        raise HTTPException(status_code=400, detail="No text available for analysis")

    # Helper to extract JSON from JSONResponse
    def extract_json(response):
        if isinstance(response, JSONResponse):
            import json
            return json.loads(response.body.decode())
        return response if isinstance(response, dict) else {}

    # Classification first (for UI step)
    try:
        cls_out = await module_classify(doc_id)
        cls_data = extract_json(cls_out)
        doc_type = cls_data.get("document_type")
    except Exception:
        doc_type = None

    # Core modules
    ner_out = await module_ner(doc_id)
    clauses_out = await module_clauses(doc_id)
    risk_out = await module_risk(doc_id)
    tl_out = await module_timeline(doc_id)

    # Extract JSON from responses
    ner_data = extract_json(ner_out)
    clauses_data = extract_json(clauses_out)
    risk_data = extract_json(risk_out)
    tl_data = extract_json(tl_out)

    # Verification + signer validation
    try:
        verification = verify_document(text)
        signers_found = ner_data.get("entities", {}).get("signers", [])
        signer_validation = validate_signers(text, signers_found)
    except Exception:
        verification = {"marker": None, "ai_confidence": None}
        signer_validation = {}

    # Combine into analytics & compute final score
    analytics = ner_data.get("stats", {})
    analytics.update({
        "document_type": doc_type,
        "clause_risk": clauses_data.get("clause_risk"),
        "jurisdiction_checks": risk_data.get("jurisdiction_ai"),
        "jurisdiction_curated": risk_data.get("jurisdiction_curated"),
        "timeline": tl_data.get("timeline"),
        "verified_marker": verification.get("marker"),
        "ai_confidence": verification.get("ai_confidence"),
        "signer_validation": signer_validation
    })

    try:
        from backend.app.services.risk_service import compute_combined_risk_index
        analytics["combined_risk_index"] = compute_combined_risk_index(analytics)
    except Exception:
        pass

    # Index for intelligent search (RAG)
    try:
        add_document({"doc_id": doc_id, "filename": doc.get("filename"), "combined_text": text})
    except Exception:
        pass

    try:
        analytics["legal_document_confidence"] = _legal_doc_confidence(text, analytics, doc_type)
    except Exception:
        pass
    save_document(doc_id, {"analytics": analytics})

    return JSONResponse({
        "doc_id": doc_id,
        "analytics": analytics,
        "results_summary": ner_data.get("entities", {}),
    })
