import os
import shutil
import uuid
import json
from typing import List
import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.config import UPLOAD_FOLDER
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service
from backend.app.services.storage_service import create_initial_document, get_document, save_document
from backend.app.routes.pipeline import run_full_pipeline

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg", "xls", "xlsx"}


def _safe_filename(name: str) -> str:
    return os.path.basename(name or "").replace(" ", "_")


def _extract_text_with_fallbacks(
    file_path: str,
    ext: str,
    ocr_lang: str = "auto",
    ocr_engine: str = None,
) -> tuple[List[str], str]:
    """
    Best-effort extraction path:
    1) requested lang/engine
    2) forced english OCR if first result is empty
    """
    effective_lang = ocr_service.resolve_ocr_lang_for_file(file_path, ext, ocr_lang)
    pages = ocr_service.extract_text_by_filetype(
        file_path,
        ext,
        ocr_lang=effective_lang,
        ocr_engine=ocr_engine,
    )
    if "\n".join(pages or []).strip():
        return pages or [], effective_lang

    # Retry with english OCR as a practical fallback for noisy/scanned inputs.
    retry_lang = "eng"
    retry_pages = ocr_service.extract_text_by_filetype(
        file_path,
        ext,
        ocr_lang=retry_lang,
        ocr_engine=ocr_engine,
    )
    if "\n".join(retry_pages or []).strip():
        return retry_pages or [], retry_lang

    return pages or [], effective_lang


@router.post("")
async def create_document(
    file: UploadFile = File(...),
    ocr_lang: str = Form("auto"),
    ocr_engine: str = Form(None),
    run_pipeline: bool = Form(True),
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
    try:
        two_step = analysis_service.classify_legal_two_step(combined_text or "", None)
        if not two_step.get("is_legal"):
            save_document(doc_id, {"analytics.document_category": "Non-Legal Document", "analytics.legality_score": 0})
        try:
            print(f"[CLASSIFY] two_step={bool(two_step.get('is_legal'))}, category={two_step.get('category')}")
        except Exception:
            pass
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
    except Exception:
        pass
    if not has_text:
        response["warning"] = (
            "Upload succeeded but text extraction returned empty content. "
            "Try a clearer scan or set TESSERACT_CMD for OCR."
        )

    response["analysis"] = {
        "document_type": "Unknown",
        "legality_score": 0,
        "combined_risk_index": 0,
    }

    # Trigger full pipeline asynchronously once upload completes
    if run_pipeline and has_text:
        try:
            asyncio.create_task(run_full_pipeline(doc_id))
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
    contact = ner_entities.get("contact_info") or {}
    clauses = (analytics.get("clause_risk") or {}).get("clauses") or []
    combined_text = doc.get("combined_text", "") or doc.get("text") or ""
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
        "uploaded_at": doc.get("upload_timestamp"),
        "file_info": doc.get("file_info", {}),
        "analytics": analytics,
        "results": doc.get("results", {}),
        "combined_text": combined_text,
        "key_facts": key_facts,
        "entities": {
            "names": ner_entities.get("names") or [],
            "organizations": ner_entities.get("organizations") or [],
            "emails": list(set((contact.get("emails") or []))),
            "phones": contact.get("phones") or contact.get("phone_details") or [],
            "dates": ner_entities.get("dates") or [],
            "signers": ner_entities.get("signers") or [],
        },
        "clauses": clauses,
        "qna_enabled": bool(combined_text),
    }


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
