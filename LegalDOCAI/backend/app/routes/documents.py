import os
import shutil
import uuid
import json
from typing import List, Optional
import asyncio
import re

from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.config import UPLOAD_FOLDER
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service
from backend.app.services.ml_service import classifier, ALL_SUPPORTED_CATEGORIES
from backend.app.services.storage_service import create_initial_document, get_document, save_document
from backend.app.routes.pipeline import run_full_pipeline, run_analysis_pipeline_from_text

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg", "xls", "xlsx"}
GENERIC_RULE_LABELS = {"Legal Document", "Legal Agreement", "Contract"}
UNKNOWN_TYPE_SET = {"UNKNOWN", "Unknown", "Other", "Other Document", ""}


@router.get("/status")
async def document_status(doc_id: str = None):
    if not doc_id:
        return {"status": "no_document"}
    doc = get_document(doc_id)
    if not doc:
        return {"status": "no_document"}
    return {"status": "ok", "doc_id": doc_id}

def _safe_filename(name: str) -> str:
    return os.path.basename(name or "").replace(" ", "_")


def _infer_type_from_filename(filename: str) -> str:
    name = (filename or "").lower()
    if not name:
        return ""
    if "pan" in name:
        return "PAN Card"
    if "aadhaar" in name or "aadhar" in name:
        return "Aadhaar Card"
    if "passport" in name:
        return "Passport"
    if "license" in name or "licence" in name or "dl" in name:
        return "Driving Licence"
    if "voter" in name or "epic" in name:
        return "Voter ID"
    if "ration" in name:
        return "Ration Card"
    if "resume" in name or "cv" in name:
        return "Resume"
    if "invoice" in name or "bill" in name:
        return "Invoice"
    if "bank" in name and "statement" in name:
        return "Bank Statement"
    if "patta" in name or "sale_deed" in name or "sale deed" in name:
        return "Land Property"
    return ""


def _ensure_doc_type_fallback(text: str, detected_type: str, filename: str = "") -> str:
    dt = str(detected_type or "").strip()
    if dt and dt not in UNKNOWN_TYPE_SET:
        return dt
    name_type = _infer_type_from_filename(filename)
    if name_type:
        return name_type
    try:
        res = classifier.predict_document_type(text or "")
        pred, _conf = res[0], res[1]
        pred = str(pred or "").strip()
        if pred and pred not in UNKNOWN_TYPE_SET and pred not in {"Generic Legal Document"}:
            return pred
    except Exception:
        pass
    return "Other Non Legal Document"


def _looks_like_non_judicial_stamp_paper(text: str) -> bool:
    raw = text or ""
    lower = raw.lower()
    tamil_present = any(0x0B80 <= ord(ch) <= 0x0BFF for ch in raw)
    stamp_primary = any(
        k in lower
        for k in [
            "non judicial",
            "non-judicial",
            "stamp paper",
            "india non judicial",
            "stamp vendor",
            "lic no",
        ]
    ) or bool(re.search(r"\b(ac|as)\s*\d{3,}", lower))
    regional_or_amount = (
        tamil_present
        or ("tamil nadu" in lower)
        or ("tamilnadu" in lower)
        or bool(re.search(r"\brs\.?\s*\d", lower))
        or ("rupees" in lower)
        or ("five thousand" in lower)
    )
    return bool(stamp_primary and regional_or_amount)


def _extract_text_with_fallbacks(
    file_path: str,
    ext: str,
    ocr_lang: str = "auto",
    ocr_engine: str = None,
) -> tuple[List[str], str]:
    """
    Best-effort extraction path:
    1) requested/resolved lang + engine
    2) multi-pass lang fallback (tam+eng/tam/eng)
    3) engine fallback (requested/default)
    """
    def _score_text(pages_list: List[str]) -> int:
        txt = "\n".join(pages_list or [])
        if not txt:
            return 0
        stripped = txt.strip()
        if not stripped:
            return 0
        tamil_chars = sum(1 for c in stripped if 0x0B80 <= ord(c) <= 0x0BFF)
        english_chars = sum(1 for c in stripped if ("a" <= c.lower() <= "z"))
        digits = sum(1 for c in stripped if c.isdigit())
        return len(stripped) + (tamil_chars * 4) + english_chars + digits

    effective_lang = ocr_service.resolve_ocr_lang_for_file(file_path, ext, ocr_lang)
    lang_candidates: List[str] = []
    for cand in [effective_lang, ocr_lang, "tam+eng", "tam", "eng"]:
        c = (cand or "").strip()
        if not c or c.lower() == "auto":
            continue
        if c not in lang_candidates:
            lang_candidates.append(c)
    if not lang_candidates:
        lang_candidates = ["eng"]

    # Try caller-requested engine first, then default.
    engine_candidates: List[Optional[str]] = []
    for eng in [ocr_engine, None]:
        if eng not in engine_candidates:
            engine_candidates.append(eng)

    best_pages: List[str] = []
    best_lang = effective_lang
    best_score = 0

    for lang in lang_candidates:
        for eng in engine_candidates:
            try:
                pages = ocr_service.extract_text_by_filetype(
                    file_path,
                    ext,
                    ocr_lang=lang,
                    ocr_engine=eng,
                ) or []
            except Exception:
                continue
            score = _score_text(pages)
            if score > best_score:
                best_pages, best_lang, best_score = pages, lang, score
            if score >= 40:
                return pages, lang

    return best_pages, best_lang


@router.post("")
async def create_document(
    file: UploadFile = File(...),
    ocr_lang: str = Form("auto"),
    ocr_engine: str = Form(None),
    run_pipeline: bool = Form(True),
    debug: bool = Form(False),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = ocr_service.detect_file_type(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: PDF, DOCX, TXT, PNG, JPG, JPEG, XLS, XLSX",
        )

    doc_id = str(uuid.uuid4())
    safe_name = _safe_filename(file.filename)
    docs_dir = os.path.join(UPLOAD_FOLDER, "documents")
    os.makedirs(docs_dir, exist_ok=True)
    stored_path = os.path.join(docs_dir, f"{doc_id}_{safe_name}")

    with open(stored_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages, effective_lang = _extract_text_with_fallbacks(
            file_path=stored_path,
            ext=ext,
            ocr_lang=ocr_lang,
            ocr_engine=ocr_engine,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {e}")

    combined_text = "\n".join(pages or [])
    has_text = bool(combined_text.strip())

    try:
        print(f"[UPLOAD] doc_id={safe_name and '...' or ''} created id, ext={ext}, stored={stored_path}")
        print(f"[OCR] lang={effective_lang}, pages={len(pages or [])}, has_text={has_text}")
    except Exception:
        pass

    create_initial_document(
        doc_id=doc_id,
        filename=safe_name,
        file_type=ext,
        text=combined_text,
        pages=pages,
    )
    try:
        save_document(doc_id, {"storage_path": stored_path})
    except Exception:
        pass
    save_document(
        doc_id,
        {
            "status": "uploaded",
            "storage_path": stored_path,
            "ocr_lang": effective_lang,
            "has_text": has_text,
            "file_info": {
                "filename": safe_name,
                "type": ext,
                "pages": len(pages or []),
            },
        },
    )
    # Deterministic fast classification immediately after OCR
    if has_text:
        try:
            det_bootstrap = analysis_service.deterministic_classify_document(combined_text or "")
            det_type = str(det_bootstrap.get("document_type") or "UNKNOWN")
            det_type = _ensure_doc_type_fallback(combined_text or "", det_type, safe_name)
            det_conf = int(str(det_bootstrap.get("final_confidence") or "0").replace("%", "") or 0)
            non_legal_types = {"Resume", "Invoice", "ID Card", "Driving Licence", "Bank Statement", "Other Non Legal Document"}
            base_payload = {
                "analytics.document_type": det_type,
                "analytics.ai_confidence": det_conf,
                "analytics.classification_source": det_bootstrap.get("classification_source"),
                "analytics.stability_hash": det_bootstrap.get("stability_hash"),
                "deterministic_classification": det_bootstrap,
            }
            if det_type in non_legal_types:
                base_payload.update({
                    "analytics.document_category": "Non-Legal Document",
                    "analytics.verified_marker": "Verified",
                    "analytics.legality_score": 0,
                    "analytics.legal_status": "NOT_LEGAL",
                })
            save_document(doc_id, base_payload)
            try:
                print(f"[CLASSIFY] deterministic type='{det_type}' ai_conf={det_conf}% source={det_bootstrap.get('classification_source')}")
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[CLASSIFY] fast detection failed: {e}")
            except Exception:
                pass

    response = {
        "doc_id": doc_id,
        "filename": safe_name,
        "status": "uploaded" if has_text else "uploaded_no_text",
        "message": "Document uploaded successfully",
    }
    try:
        print(f"[UPLOAD] text_length={len(combined_text or '')} pages={len(pages or [])} has_text={has_text}")
        preview = (combined_text or "")[:200].replace("\n"," ")
        print(f"[UPLOAD] text_preview: {preview}")
    except Exception:
        pass
    if not has_text:
        response["warning"] = (
            "Upload succeeded but text extraction returned empty content. "
            "Try a clearer scan or set TESSERACT_CMD for OCR."
        )
        response["type"] = "Unknown"
        response["confidence"] = 0
        response["error"] = "Text extraction failed"

    # Real analysis output (no placeholders)
    if has_text:
        try:
            det_result = analysis_service.deterministic_classify_document(combined_text or "")
            det_type = str(det_result.get("document_type") or "UNKNOWN")
            if _looks_like_non_judicial_stamp_paper(combined_text or ""):
                det_type = "Non Judicial Stamp Paper"
            det_type = _ensure_doc_type_fallback(combined_text or "", det_type, safe_name)
            det_conf = int(str(det_result.get("final_confidence") or "0").replace("%", "") or 0)
            save_document(doc_id, {
                "analytics.document_type": det_type,
                "analytics.ai_confidence": det_conf,
                "analytics.classification_source": det_result.get("classification_source"),
                "analytics.stability_hash": det_result.get("stability_hash"),
                "deterministic_classification": det_result,
                "analytics.legal_status": "UNKNOWN",
                "analytics.legal_status_text": "Needs review"
            })
            response["document_name"] = safe_name
            response["type"] = det_type
            response["legal_status"] = "Needs review"
            response["confidence"] = det_conf
            response["deterministic_classification"] = det_result
            response["analysis"] = {
                "document_type": det_type,
                "confidence": det_conf,
                "legal_status": "Needs review",
            }
            try:
                print(f"[FAST-DETERMINISTIC] type={det_type} conf={det_conf}% source={det_result.get('classification_source')}")
            except Exception:
                pass
        except Exception as e:
            response["analysis"] = {
                "document_type": locals().get("det_type") or "Unknown",
                "confidence": 70,
                "legal_status": "Needs review",
            }
            try:
                print(f"[FAST] pipeline failed: {e}")
            except Exception:
                pass
    else:
        fallback_type = None
        try:
            name_only = (safe_name or "").lower()
            # Try filename-based rule detection (e.g., resume.pdf, licence.jpg)
            fallback_type = classifier.canonicalize_category(classifier.detect_document_type_by_rules(name_only))
            if fallback_type in ALL_SUPPORTED_CATEGORIES and fallback_type not in {"Other Indian Legal Document", "Other Non Legal Document"}:
                response["analysis"] = {
                    "document_type": fallback_type,
                    "confidence": 90,
                    "legal_status": "Needs review",
                }
                response["type"] = fallback_type
                response["confidence"] = 90
                response["error"] = "Text extraction failed (filename-based classification applied)"
            else:
                response["analysis"] = {
                    "document_type": "Unknown",
                    "confidence": 0,
                    "legal_status": "Needs review",
                }
                response["type"] = "Unknown"
                response["confidence"] = 0
                response["error"] = "Text extraction failed"
        except Exception:
            response["analysis"] = {
                "document_type": "Unknown",
                "confidence": 0,
                "legal_status": "Needs review",
            }
            response["type"] = "Unknown"
            response["confidence"] = 0
            response["error"] = "Text extraction failed"

    # Trigger full pipeline asynchronously once upload completes
    if run_pipeline and has_text:
        try:
            asyncio.create_task(run_full_pipeline(doc_id, debug=debug))
            save_document(doc_id, {"pipeline_status": "queued"})
            try:
                print(f"[PIPELINE] queued doc_id={doc_id}")
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[PIPELINE] trigger failed for doc_id={doc_id}: {e}")
            except Exception:
                pass

    return JSONResponse(response)

@router.post("/{doc_id}/reclassify")
async def reclassify_document(doc_id: str, ocr_lang: str = Form(None), ocr_engine: str = Form(None)):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = _safe_filename(doc.get("filename") or "")
    ext = doc.get("file_info", {}).get("type") or ocr_service.detect_file_type(filename)
    stored_path = doc.get("storage_path") or os.path.join(UPLOAD_FOLDER, "documents", f"{doc_id}_{filename}")
    if not stored_path or not os.path.exists(stored_path):
        raise HTTPException(status_code=400, detail="Stored file path missing")
    try:
        eff = (ocr_lang or "auto")
        eng = (ocr_engine or None)
        pages, eff_lang = _extract_text_with_fallbacks(stored_path, ext, ocr_lang=eff, ocr_engine=eng)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")
    text = "\n".join(pages or [])
    save_document(doc_id, {"combined_text": text, "pages": pages})
    try:
        try:
            results, analytics, _ = await analysis_service.analyze_text_overall_async(text or "")
            ents = (results or {}).get("entities") or {}
            names = ents.get("names") or []
            orgs = ents.get("organizations") or []
            save_document(doc_id, {
                "pipeline.ner.entities": {"names": names, "organizations": orgs},
                "analytics.names": names,
                "analytics.organizations": orgs,
            })
        except Exception:
            pass
        det = analysis_service.deterministic_classify_document(text or "")
        det_type = str(det.get("document_type") or "Unknown")
        det_conf = int(str(det.get("final_confidence") or "0").replace("%", "") or 0)
        lower = (text or "").lower()
        property_types = {"Land Property","Non Judicial Stamp Paper","Sale Deed","Patta","TSLR","Encumbrance Certificate"}
        amount_cue = (
            ("rs." in lower)
            or ("rs " in lower)
            or bool(re.search(r"\brs\.?\s*\d", lower))
            or ("rupees" in lower)
            or ("five thousand rupees" in lower)
        )
        if _looks_like_non_judicial_stamp_paper(text or "") or (det_type in property_types and amount_cue):
            det_type = "Non Judicial Stamp Paper"
        det_type = _ensure_doc_type_fallback(text or "", det_type, filename)
        ver = classifier.verify_document_ml(text or "")
        marker = str(ver.get("marker") or "").title()
        ai_conf = int(ver.get("ai_confidence") or det_conf or 0)
        legal_status = analysis_service.derive_legal_status({"document_type": det_type, "ai_confidence": ai_conf, "verified_marker": marker}, ver)
        save_document(doc_id, {
            "analytics.document_type": det_type,
            "analytics.ai_confidence": ai_conf,
            "analytics.verified_marker": marker,
            "analytics.legal_status": legal_status.get("status"),
            "analytics.status_summary": legal_status.get("summary"),
            "analytics.classification_source": det.get("classification_source"),
            "analytics.stability_hash": det.get("stability_hash"),
            "deterministic_classification": det,
            "ml_verification": ver,
        })
        return {"doc_id": doc_id, "type": det_type, "confidence": ai_conf, "marker": marker, "ocr_lang": eff_lang}
    except Exception as e:
        return {"doc_id": doc_id, "type": "Unknown", "confidence": 0, "error": str(e)}

@router.post("/{doc_id}/reclassify/tamil")
async def reclassify_document_tamil(doc_id: str, ocr_engine: str = Form(None)):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    filename = _safe_filename(doc.get("filename") or "")
    ext = doc.get("file_info", {}).get("type") or ocr_service.detect_file_type(filename)
    stored_path = doc.get("storage_path") or os.path.join(UPLOAD_FOLDER, "documents", f"{doc_id}_{filename}")
    if not stored_path or not os.path.exists(stored_path):
        raise HTTPException(status_code=400, detail="Stored file path missing")
    try:
        pages, eff_lang = _extract_text_with_fallbacks(stored_path, ext, ocr_lang="tam+eng", ocr_engine=(ocr_engine or None))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")
    text = "\n".join(pages or [])
    save_document(doc_id, {"combined_text": text, "pages": pages})
    try:
        try:
            results, analytics, _ = await analysis_service.analyze_text_overall_async(text or "")
            ents = (results or {}).get("entities") or {}
            names = ents.get("names") or []
            orgs = ents.get("organizations") or []
            save_document(doc_id, {
                "pipeline.ner.entities": {"names": names, "organizations": orgs},
                "analytics.names": names,
                "analytics.organizations": orgs,
            })
        except Exception:
            pass
        det = analysis_service.deterministic_classify_document(text or "")
        det_type = str(det.get("document_type") or "Unknown")
        det_conf = int(str(det.get("final_confidence") or "0").replace("%", "") or 0)
        lower = (text or "").lower()
        property_types = {"Land Property","Non Judicial Stamp Paper","Sale Deed","Patta","TSLR","Encumbrance Certificate"}
        amount_cue = (
            ("rs." in lower)
            or ("rs " in lower)
            or bool(re.search(r"\brs\.?\s*\d", lower))
            or ("rupees" in lower)
            or ("five thousand rupees" in lower)
        )
        if _looks_like_non_judicial_stamp_paper(text or "") or (det_type in property_types and amount_cue):
            det_type = "Non Judicial Stamp Paper"
        det_type = _ensure_doc_type_fallback(text or "", det_type, filename)
        ver = classifier.verify_document_ml(text or "")
        marker = str(ver.get("marker") or "").title()
        ai_conf = int(ver.get("ai_confidence") or det_conf or 0)
        legal_status = analysis_service.derive_legal_status({"document_type": det_type, "ai_confidence": ai_conf, "verified_marker": marker}, ver)
        save_document(doc_id, {
            "analytics.document_type": det_type,
            "analytics.ai_confidence": ai_conf,
            "analytics.verified_marker": marker,
            "analytics.legal_status": legal_status.get("status"),
            "analytics.status_summary": legal_status.get("summary"),
            "analytics.classification_source": det.get("classification_source"),
            "analytics.stability_hash": det.get("stability_hash"),
            "deterministic_classification": det,
            "ml_verification": ver,
        })
        return {"doc_id": doc_id, "type": det_type, "confidence": ai_conf, "marker": marker, "ocr_lang": eff_lang}
    except Exception as e:
        return {"doc_id": doc_id, "type": "Unknown", "confidence": 0, "error": str(e)}

@router.post("/{doc_id}/refresh-entities")
async def refresh_entities(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    text = (doc.get("combined_text") or doc.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text available for entity refresh")
    try:
        results, analytics, _ = await analysis_service.analyze_text_overall_async(text)
        ents = (results or {}).get("entities") or {}
        names = ents.get("names") or []
        orgs = ents.get("organizations") or []
        save_document(doc_id, {
            "pipeline.ner.entities": {"names": names, "organizations": orgs},
            "analytics.names": names,
            "analytics.organizations": orgs,
        })
        return {"doc_id": doc_id, "names": names, "organizations": orgs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity refresh failed: {e}")

@router.post("/batch")
async def create_documents_batch(
    files: List[UploadFile] = File(...),
    ocr_lang: str = Form("auto"),
    ocr_engine: str = Form(None),
    run_pipeline: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")

    results = []
    succeeded = 0
    failed = 0
    for f in files:
        try:
            res = await create_document(
                file=f,
                ocr_lang=ocr_lang,
                ocr_engine=ocr_engine,
                run_pipeline=run_pipeline,
            )
            body = json.loads(res.body.decode("utf-8")) if isinstance(res, JSONResponse) else {}
            body["ok"] = True
            results.append(body)
            succeeded += 1
        except HTTPException as e:
            failed += 1
            results.append(
                {
                    "ok": False,
                    "filename": f.filename,
                    "status_code": e.status_code,
                    "error": e.detail,
                }
            )
        except Exception as e:
            failed += 1
            results.append(
                {
                    "ok": False,
                    "filename": f.filename,
                    "status_code": 500,
                    "error": str(e),
                }
            )

    return {
        "total": len(files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@router.get("/{doc_id}")
async def get_document_by_id(doc_id: str = Path(..., description="Document ID")):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.pop("_id", None)
    analytics = doc.get("analytics", {}) or {}
    pipeline = doc.get("pipeline", {}) or {}
    ner_block = pipeline.get("ner", {}) or {}
    ner_entities = ner_block.get("entities", {}) or {}
    nested_entities = ner_entities.get("entities") or {}
    contact = ner_entities.get("contact_info") or {}
    names = ner_entities.get("names") or nested_entities.get("names") or []
    organizations = ner_entities.get("organizations") or nested_entities.get("organizations") or []
    emails = ner_entities.get("emails") or contact.get("emails") or []
    raw_phones = ner_entities.get("phones") or contact.get("phones") or contact.get("phone_details") or []
    phones = []
    for p in (raw_phones or []):
        if isinstance(p, dict):
            val = p.get("normalized") or p.get("number")
            if val:
                phones.append(str(val))
        elif p:
            phones.append(str(p))
    phones = list(dict.fromkeys(phones))
    signers = ner_entities.get("signers") or []
    dates = ner_entities.get("dates") or []
    clauses = (analytics.get("clause_risk") or {}).get("clauses") or []
    combined_text = doc.get("combined_text", "") or doc.get("text") or ""
    if (not names) and combined_text:
        try:
            from backend.app.services.analysis_service import answer_with_fallback
            aa = answer_with_fallback("name", combined_text)
            if aa and isinstance(aa, str) and ("No clear person names" not in aa):
                names = [s.strip() for s in aa.split(",") if s.strip()]
        except Exception:
            pass
    try:
        from backend.app.services.analysis_service import sanitize_names, sanitize_orgs
        names = sanitize_names(names or [])
        organizations = sanitize_orgs(organizations or [])
    except Exception:
        pass
    try:
        key_facts = analysis_service.extract_key_facts_regex(combined_text or "")
        key_facts["summary"] = analytics.get("summary")
    except Exception:
        key_facts = {"summary": analytics.get("summary")}
    try:
        print(f"[API] get_document_by_id doc_id={doc_id} entities={len(ner_entities.get('names', [])) + len(ner_entities.get('organizations', []))} clauses={len(clauses)}")
    except Exception:
        pass
    return {
        "doc_id": doc.get("doc_id"),
        "filename": doc.get("filename"),
        "status": doc.get("status", "unknown"),
        "pipeline_status": doc.get("pipeline_status"),
        "uploaded_at": doc.get("upload_timestamp"),
        "file_info": doc.get("file_info", {}),
        "analytics": analytics,
        "deterministic_classification": doc.get("deterministic_classification") or {},
        "results": doc.get("results", {}),
        "combined_text": combined_text,
        "key_facts": key_facts,
        "entities": {
            "names": names,
            "organizations": organizations,
            "emails": list(dict.fromkeys([str(e) for e in (emails or []) if e])),
            "phones": phones,
            "dates": dates,
            "signers": signers,
            "parties": list(dict.fromkeys([*(names or []), *(organizations or [])])),
        },
        "clauses": clauses,
        "qna_enabled": bool(combined_text),
    }


@router.get("/{doc_id}/debug")
async def get_document_debug(doc_id: str = Path(..., description="Document ID")):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    debug_payload = doc.get("debug")
    if not debug_payload:
        return {"doc_id": doc_id, "debug": None, "message": "No debug payload stored"}
    return {"doc_id": doc_id, "debug": debug_payload}


@router.get("/{doc_id}/export/json")
async def export_document_json(doc_id: str = Path(..., description="Document ID")):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.pop("_id", None)
    return JSONResponse(
        content=doc,
        headers={"Content-Disposition": f'attachment; filename="document_{doc_id}.json"'},
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str = Path(..., description="Document ID")):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_path = doc.get("storage_path")
    result = collection.delete_one({"doc_id": doc_id})
    if not getattr(result, "deleted_count", 0):
        raise HTTPException(status_code=500, detail="Failed to delete document")

    if storage_path and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except Exception:
            pass

    return {"doc_id": doc_id, "status": "deleted"}
