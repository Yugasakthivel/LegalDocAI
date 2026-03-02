from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from backend.app.core.config import UPLOAD_FOLDER
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service
from backend.app.routes.pipeline import run_full_pipeline
from backend.app.core.deps import get_current_user
from bson import ObjectId
import os
import shutil
import time
import json
import re

router = APIRouter(tags=["upload"])

def _sse(d):
    try: return f"data: {json.dumps(d)}\n\n"
    except Exception: return "data: {}\n\n"

async def _run_pipeline_compat(doc_id: str, debug: bool = False):
    # Backward-compat: allow monkeypatched/legacy run_full_pipeline(doc_id)
    # while supporting the newer run_full_pipeline(doc_id, debug=...).
    try:
        return await run_full_pipeline(doc_id, debug=debug)
    except TypeError:
        return await run_full_pipeline(doc_id)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    ocr_lang: str = Form(None),
    ocr_engine: str = Form(None),
    debug: bool = Form(False),
    user: dict = Depends(get_current_user),
):
    """Upload, scan, analyze, verify and save document"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    
    # 1. Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    # 2. Detect File Type
    file_type = ocr_service.detect_file_type(file.filename)
    eff_lang = ocr_service.resolve_ocr_lang_for_file(file_path, file_type, ocr_lang or "auto")
    
    # 3. OCR Extraction
    try:
        page_texts = ocr_service.extract_text_by_filetype(file_path, file_type, ocr_lang=eff_lang, ocr_engine=ocr_engine)
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Failed to extract text: {e}")

    if not any(page_texts):
        raise HTTPException(status_code=400,detail="Failed to extract text from file.")

    full_text = "\n".join(page_texts)

    # 4. Create Initial Document
    doc_id = str(ObjectId())
    doc_data = {
        "doc_id": doc_id,
        "filename": file.filename,
        "file_path": file_path,
        "results": [], # To be filled by pipeline
        "analytics": {}, # To be filled by pipeline
        "combined_text": full_text,
        "file_type": file_type,
        "ocr_lang": eff_lang
    }
    collection.insert_one(doc_data)

    # 5. Run Full Pipeline
    pipeline_result = await _run_pipeline_compat(doc_id, debug=debug)
    
    return JSONResponse(pipeline_result)

@router.post("/upload-stream")
async def upload_stream(
    file: UploadFile = File(...),
    ocr_lang: str = Form(None),
    ocr_engine: str = Form(None),
    debug: bool = Form(False),
    user: dict = Depends(get_current_user),
):
    """Upload with live server-side progress via text/event-stream"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    
    async def generate():
        t_total_start = time.perf_counter()
        yield _sse({"stage":"start","message":"Starting upload"})
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path,"wb") as f:
            shutil.copyfileobj(file.file,f)
        
        yield _sse({"stage":"saved"})
        
        file_type = ocr_service.detect_file_type(file.filename)
        eff_lang = ocr_service.resolve_ocr_lang_for_file(file_path, file_type, ocr_lang or "auto")
        
        yield _sse({"stage":"lang_resolved","ocr_lang": eff_lang})
        
        try:
            page_texts = ocr_service.extract_text_by_filetype(file_path, file_type, ocr_lang=eff_lang, ocr_engine=ocr_engine)
        except Exception as e:
            yield _sse({"stage":"error","detail": f"Failed to extract text: {e}"})
            return
            
        if not any(page_texts):
            yield _sse({"stage":"error","detail":"Failed to extract text from file."})
            return
            
        yield _sse({"stage":"extracted","pages": len(page_texts)})
        full_text = "\n".join(page_texts)
        
        doc_id = str(ObjectId())
        doc_data = {
            "doc_id": doc_id,
            "filename": file.filename,
            "file_path": file_path,
            "results": [],
            "combined_text": full_text
        }
        collection.insert_one(doc_data)
        
        yield _sse({"stage":"analyzing", "message": "Running full analysis pipeline..."})
        
        # Run Pipeline
        pipeline_result = await _run_pipeline_compat(doc_id, debug=debug)
        analytics = pipeline_result.get("analytics", {})
        debug_payload = pipeline_result.get("_debug")
        
        yield _sse({"stage":"analyzed","legality_score":analytics.get("legality_score")})
        yield _sse({
            "stage":"verified",
            "marker":analytics.get("verified_marker"),
            "confidence":analytics.get("ai_confidence"),
            "anomaly_detected": analytics.get("anomaly_detected"),
            "anomaly_reason": analytics.get("anomaly_reason"),
            "fraud_flag": analytics.get("fraud_flag"),
            "fraud_reason": analytics.get("fraud_reason"),
        })
        done_payload = {"stage": "done", "doc_id": doc_id}
        if debug_payload:
            done_payload["debug"] = debug_payload
        yield _sse(done_payload)

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/ocr-clean")
async def ocr_clean(
    file: UploadFile = File(...),
    ocr_lang: str = Form(None),
    ocr_engine: str = Form(None)
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    ext = ocr_service.detect_file_type(file.filename)
    tmp_path = os.path.join(UPLOAD_FOLDER, f"ocrc_{file.filename}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        eff_lang = ocr_service.resolve_ocr_lang_for_file(tmp_path, ext, ocr_lang or "auto")
        if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
            eff_lang = "eng"
        
        pages_structured = ocr_service.extract_text_by_filetype(tmp_path, ext, ocr_lang=eff_lang, ocr_engine=ocr_engine)
        
        cleaned_pages = []
        for t in pages_structured:
            t0 = re.sub(r"\r\n", "\n", t or "")
            t1 = re.sub(r"[ \t]+\n", "\n", t0)
            t2 = re.sub(r"\n{3,}", "\n\n", t1)
            cleaned_pages.append(t2.strip())
        combined = "\n\n---\n\n".join(cleaned_pages)
        return {"pages": cleaned_pages, "combined_text": combined, "ocr_lang": eff_lang}
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
