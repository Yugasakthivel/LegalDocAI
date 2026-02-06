from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from backend.app.core.config import OPENAI_MODEL, UPLOAD_FOLDER
from backend.app.core.deps import get_openai_client, get_current_user, update_openai_key, is_openai_ready
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service
from backend.app.services.ml_service import classifier
from backend.app import vectorstore
import time
import random
import re
import os
import shutil
import uuid
from typing import Optional
_prefer_local_translation = os.getenv("USE_LOCAL_TRANSLATION_FIRST", "0").strip() == "1"

router = APIRouter(tags=["ai"])

@router.get("/openai-status")
async def openai_status():
    return {"ready": is_openai_ready()}

@router.get("/functions")
async def list_functions():
    return {
        "functions": [
            {"name": "ai_response", "params": ["text", "question"]},
            {"name": "chat_rag", "params": ["doc_id", "question"]},
            {"name": "start_analysis", "params": ["email", "uploadedDocumentId"]},
            {"name": "translate_analyze", "params": ["file", "target_lang", "ocr_lang"]}
        ]
    }

@router.post("/update-key")
async def update_key(key: str = Form(...), user: dict = Depends(get_current_user)):
    """Update OpenAI API key at runtime"""
    success = update_openai_key(key)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update API key")
    return {"status": "success", "message": "API key updated successfully"}

class AIRequest(BaseModel):
    text: str
    question: str = None

class StartRequest(BaseModel):
    email: str
    uploadedDocumentId: str | None = None

def _estimate_tokens(text: str, max_tokens: int) -> int:
    approx_prompt = max(1, len(text) // 4)
    return approx_prompt + max_tokens

def _parse_retry_after_seconds(message: str) -> float:
    m = re.search(r"Please try again in\s+(\d+)(?:\.(\d+))?s", message)
    if m:
        whole = int(m.group(1))
        frac = m.group(2)
        return whole + (float(f"0.{frac}") if frac else 0.0)
    return 0.0

@router.post("/ai-response")
async def ai_response(req: AIRequest, user: dict = Depends(get_current_user)):
    """Ask custom questions or summarize document text"""
    prompt = (
        f"{req.text}\n\nQuestion: {req.question}\n"
        "Answer using only the extracted document data above. "
        "If the information is not present, reply exactly: Not available in the document."
    ) if req.question else (
        f"Summarize the following document text briefly using non-technical language:\n\n{req.text}"
    )
    try:
        def _delay(base, attempt, jitter=0.3, cap=8.0):
            d = min(base * (2 ** attempt), cap)
            j = d * jitter
            return d + random.uniform(-j, j)
        attempt = 0
        max_toks = 300
        while True:
            try:
                response = get_openai_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.3,
                    max_tokens=max_toks
                )
                break
            except Exception as e:
                sc = getattr(e, "status_code", None)
                if sc == 429 or "429" in str(e):
                    if attempt >= 4:
                        raise
                    ra = _parse_retry_after_seconds(str(e))
                    if ra > 0:
                        time.sleep(ra + random.uniform(0.05, 0.25))
                    else:
                        time.sleep(_delay(0.5, attempt))
                    attempt += 1
                    continue
                raise
        return {"answer":response.choices[0].message.content}
    except Exception as e:
        try:
            if req.question:
                ans = analysis_service.answer_with_fallback(req.question, req.text or "")
            else:
                ans = analysis_service.summarize_fallback(req.text or "")
            return {"answer": ans}
        except Exception:
            return {"answer":f"⚠️ Failed to call OpenAI: {e}"}

@router.post("/chat-rag")
async def chat_rag(doc_id: str, question: str, user: dict = Depends(get_current_user)):
    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="doc_id and question are required.")
    doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "filename": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    # Ensure document is indexed
    if doc.get("combined_text"):
        vectorstore.add_document({"doc_id": doc_id, "filename": doc.get("filename",""), "combined_text": doc["combined_text"]})
        
    retrieved = vectorstore.search(question, top_k=3)
    contexts = []
    for r in retrieved:
        # If vectorstore returns doc_ids from other docs, filter or use?
        # main.py logic seems to assume search index contains *all* docs.
        # But if we want RAG over *this* doc, we should filter.
        # However, main.py search_index searched ALL docs.
        d_res = collection.find_one({"doc_id": r["doc_id"]}, {"_id": 0, "combined_text": 1})
        if d_res and d_res.get("combined_text"):
            contexts.append(d_res["combined_text"][:2000])
            
    context_blob = "\n\n".join(contexts) if contexts else doc.get("combined_text","")[:2000]
    try:
        prompt = (
            "Answer the question using the provided document context. "
            "Cite relevant snippets. If not found, say not explicitly stated.\n\n"
            "Context:\n" + context_blob + "\n\nQuestion:\n" + question
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=600
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        try:
            answer = analysis_service.answer_with_fallback(question, context_blob)
        except Exception:
            answer = f"Failed to answer: {e}"
    return {"answer": answer, "sources": retrieved}

@router.post("/exec")
async def exec_function(
    function: str = Form(...),
    text: str = Form(None),
    question: str = Form(None),
    doc_id: str = Form(None),
    email: str = Form(None),
    uploadedDocumentId: str = Form(None),
    user: dict = Depends(get_current_user)
):
    fname = (function or "").strip().lower()
    if fname == "ai_response":
        req = AIRequest(text=text or "", question=question)
        return await ai_response(req, user)
    if fname == "chat_rag":
        if not doc_id or not question:
            raise HTTPException(status_code=400, detail="doc_id and question are required")
        return await chat_rag(doc_id, question, user)
    if fname == "start_analysis":
        req = StartRequest(email=email or "", uploadedDocumentId=uploadedDocumentId)
        return await start_analysis(req)
    if fname == "translate_analyze":
        raise HTTPException(status_code=400, detail="Use /translate-analyze for file uploads")
    raise HTTPException(status_code=400, detail="Unknown function name")

@router.post("/ml/train")
async def ml_train(
    dataset_json: str = Form(None),
    dataset_url: str = Form(None),
    dataset_file: UploadFile = File(None),
    user: dict = Depends(get_current_user)
):
    """
    Train the local ML classifier with provided dataset.
    Accepts:
    - dataset_json: JSON string like [{"text":"...","label":"..."}]
    - dataset_url: HTTP URL returning JSON in the same format
    - dataset_file: uploaded file (CSV with headers text,label or JSON array)
    """
    samples = []
    try:
        if dataset_json:
            import json
            arr = json.loads(dataset_json)
            if isinstance(arr, list):
                for it in arr:
                    t = (it or {}).get("text") or ""
                    l = (it or {}).get("label") or ""
                    if t and l:
                        samples.append((t, l))
        elif dataset_url:
            try:
                import httpx
                r = httpx.get(dataset_url, timeout=10)
                r.raise_for_status()
                import json
                arr = r.json()
                if isinstance(arr, list):
                    for it in arr:
                        t = (it or {}).get("text") or ""
                        l = (it or {}).get("label") or ""
                        if t and l:
                            samples.append((t, l))
            except Exception:
                pass
        elif dataset_file:
            name = (dataset_file.filename or "").lower()
            buf = await dataset_file.read()
            if name.endswith(".json"):
                import json
                try:
                    arr = json.loads(buf.decode("utf-8", errors="ignore"))
                    if isinstance(arr, list):
                        for it in arr:
                            t = (it or {}).get("text") or ""
                            l = (it or {}).get("label") or ""
                            if t and l:
                                samples.append((t, l))
                except Exception:
                    pass
            else:
                # Treat as CSV: headers text,label
                import csv, io
                try:
                    f = io.StringIO(buf.decode("utf-8", errors="ignore"))
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        t = (row or {}).get("text") or ""
                        l = (row or {}).get("label") or ""
                        if t and l:
                            samples.append((t, l))
                except Exception:
                    pass
    except Exception:
        samples = []

    texts = [t for (t, _) in samples]
    labels = [l for (_, l) in samples]
    res = classifier.train_model(texts, labels)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error") or "Training failed")
    return {"status": "ok", "trained": len(texts), "details": res.get("details")}

@router.post("/ml/evaluate")
async def ml_evaluate(
    dataset_json: str = Form(None),
    dataset_url: str = Form(None),
    dataset_file: UploadFile = File(None),
    user: dict = Depends(get_current_user)
):
    samples = []
    try:
        if dataset_json:
            import json
            arr = json.loads(dataset_json)
            if isinstance(arr, list):
                for it in arr:
                    t = (it or {}).get("text") or ""
                    l = (it or {}).get("label") or ""
                    if t and l:
                        samples.append((t, l))
        elif dataset_url:
            try:
                import httpx
                r = httpx.get(dataset_url, timeout=20)
                r.raise_for_status()
                arr = r.json()
                if isinstance(arr, list):
                    for it in arr:
                        t = (it or {}).get("text") or ""
                        l = (it or {}).get("label") or ""
                        if t and l:
                            samples.append((t, l))
            except Exception:
                pass
        elif dataset_file:
            name = (dataset_file.filename or "").lower()
            buf = await dataset_file.read()
            if name.endswith(".json"):
                import json
                try:
                    arr = json.loads(buf.decode("utf-8", errors="ignore"))
                    if isinstance(arr, list):
                        for it in arr:
                            t = (it or {}).get("text") or ""
                            l = (it or {}).get("label") or ""
                            if t and l:
                                samples.append((t, l))
                except Exception:
                    pass
            else:
                import csv, io
                try:
                    f = io.StringIO(buf.decode("utf-8", errors="ignore"))
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        t = (row or {}).get("text") or ""
                        l = (row or {}).get("label") or ""
                        if t and l:
                            samples.append((t, l))
                except Exception:
                    pass
    except Exception:
        samples = []
    texts = [t for (t, _) in samples]
    labels = [l for (_, l) in samples]
    res = classifier.evaluate_dataset(texts, labels)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error") or "Evaluation failed")
    return {"status": "ok", "evaluated": len(texts), "metrics": {"accuracy": res.get("accuracy"), "report": res.get("report"), "confusion_matrix": res.get("confusion_matrix")}}

@router.post("/translate-analyze")
async def translate_analyze(file: UploadFile = File(...), target_lang: str = Form("English"), ocr_lang: str = Form(None), user: dict = Depends(get_current_user)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    path = os.path.join(UPLOAD_FOLDER, f"trans_{uuid.uuid4()}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ext = ocr_service.detect_file_type(file.filename)
    warnings = []
    eff_lang = ocr_service.resolve_ocr_lang_for_file(path, ext, ocr_lang or "auto")
    if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
        warnings.append("Tamil OCR not installed; using English OCR.")
        eff_lang = "eng"
    pages = ocr_service.extract_text_by_filetype(path, ext, ocr_lang=eff_lang)
    src_text = "\n".join(pages)
    if not src_text.strip():
        warnings.append("OCR could not extract any text from the document.")

    try:
        if not src_text.strip():
            translated = ""
        elif _prefer_local_translation or not is_openai_ready():
            def _detect_script(text: str) -> str:
                for ch in text:
                    o = ord(ch)
                    if 0x0B80 <= o <= 0x0BFF:
                        return "tam"
                    if 0x0900 <= o <= 0x097F:
                        return "hin"
                return "eng"
            def _marian_model_for(target: str, source_hint: str) -> Optional[str]:
                t = (target or "").lower()
                s = (source_hint or "").lower()
                if t.startswith("english"):
                    if "tam" in s:
                        return "Helsinki-NLP/opus-mt-ta-en"
                    if "hin" in s:
                        return "Helsinki-NLP/opus-mt-hi-en"
                return None
            def _translate_marian(text: str, target: str, source_hint: str) -> Optional[str]:
                model_name = _marian_model_for(target, source_hint)
                if not model_name:
                    auto = _detect_script(text)
                    model_name = _marian_model_for(target, auto)
                if not model_name:
                    return None
                try:
                    from transformers import MarianMTModel, MarianTokenizer
                except Exception:
                    return None
                tok = MarianTokenizer.from_pretrained(model_name)
                mdl = MarianMTModel.from_pretrained(model_name)
                chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
                outs = []
                for c in chunks:
                    enc = tok([c], return_tensors="pt", padding=True, truncation=True)
                    gen = mdl.generate(**enc, max_length=900)
                    outs.append(tok.decode(gen[0], skip_special_tokens=True))
                return "\n".join(outs).strip()
            fb = _translate_marian(src_text, target_lang, eff_lang)
            translated = (fb or src_text)
            if fb is None:
                warnings.append("Local translator unavailable; showing original text.")
        else:
            prompt = (
                f"Translate to {target_lang} preserving legal meaning and structure. "
                "Return only translated text.\n\n" + src_text[:6000]
            )
            attempt = 0
            start = time.perf_counter()
            translated = None
            while True:
                try:
                    resp = get_openai_client().chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role":"user","content":prompt}],
                        temperature=0,
                        max_tokens=800,
                        timeout=20
                    )
                    translated = (resp.choices[0].message.content or "").strip()
                    break
                except Exception as e:
                    sc = getattr(e, "status_code", None)
                    if sc == 429 or "429" in str(e):
                        if attempt >= 3:
                            raise
                        ra = _parse_retry_after_seconds(str(e))
                        if ra > 0:
                            time.sleep(ra + random.uniform(0.05, 0.25))
                        else:
                            d = min(0.5 * (2 ** attempt), 8.0)
                            j = d * 0.3
                            time.sleep(d + random.uniform(-j, j))
                        attempt += 1
                        if time.perf_counter() - start > 30:
                            raise TimeoutError("Translation timed out")
                        continue
                    raise
    except Exception as e:
        def _detect_script(text: str) -> str:
            for ch in text:
                o = ord(ch)
                if 0x0B80 <= o <= 0x0BFF:
                    return "tam"
                if 0x0900 <= o <= 0x097F:
                    return "hin"
            return "eng"
        def _marian_model_for(target: str, source_hint: str) -> Optional[str]:
            t = (target or "").lower()
            s = (source_hint or "").lower()
            if t.startswith("english"):
                if "tam" in s:
                    return "Helsinki-NLP/opus-mt-ta-en"
                if "hin" in s:
                    return "Helsinki-NLP/opus-mt-hi-en"
            return None
        def _translate_marian(text: str, target: str, source_hint: str) -> Optional[str]:
            model_name = _marian_model_for(target, source_hint)
            if not model_name:
                auto = _detect_script(text)
                model_name = _marian_model_for(target, auto)
            if not model_name:
                return None
            try:
                from transformers import MarianMTModel, MarianTokenizer
            except Exception:
                return None
            tok = MarianTokenizer.from_pretrained(model_name)
            mdl = MarianMTModel.from_pretrained(model_name)
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            outs = []
            for c in chunks:
                enc = tok([c], return_tensors="pt", padding=True, truncation=True)
                gen = mdl.generate(**enc, max_length=900)
                outs.append(tok.decode(gen[0], skip_special_tokens=True))
            return "\n".join(outs).strip()
        fb = None
        try:
            fb = _translate_marian(src_text, target_lang, eff_lang)
        except Exception:
            fb = None
        translated = fb or src_text
        msg_raw = str(e)
        msg = msg_raw.lower()
        if "insufficient_quota" in msg or "you exceeded your current quota" in msg or "openai_quota_exhausted" in msg:
            warnings.append("OpenAI quota exceeded. Using local translator if available.")
        elif "openai_not_configured" in msg or "api key" in msg or "authentication" in msg:
            warnings.append("OpenAI API key missing or invalid. Using local translator if available.")
        elif "timed out" in msg or "timeout" in msg:
            warnings.append("Translation timed out. Using local translator if available.")
        else:
            warnings.append(f"Translation failed: {str(e)}. Using local translator if available.")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    
    rs, an, working_text = analysis_service.analyze_text_overall(translated)
    base_text = working_text or translated
    an["timeline"] = analysis_service.build_timeline(base_text)
    an["document_type"] = analysis_service.classify_document_type(base_text)
    an["clause_risk"] = analysis_service.clause_risk_detection(base_text, translated_text=working_text)
    an["signer_validation"] = analysis_service.validate_signers(base_text, rs.get("signers", []))
    an["jurisdiction_checks"] = analysis_service.jurisdiction_specific_checks(base_text)
    an["jurisdiction_curated"] = analysis_service.curated_jurisdiction_templates(base_text)
    an["registration_analysis"] = analysis_service.analyze_registration(base_text)
    an["combined_risk_index"] = analysis_service.compute_combined_risk_index(an)
    if warnings:
        an["warnings"] = warnings
    return {"translated_text": base_text, "results": rs, "analytics": an}

@router.post("/start-analysis")
async def start_analysis(req: StartRequest):
    doc_id = (req.uploadedDocumentId or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="uploadedDocumentId required")
    doc = collection.find_one({"doc_id": doc_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"docId": doc_id, "documentId": doc_id, "uploadedDocumentId": doc_id}
