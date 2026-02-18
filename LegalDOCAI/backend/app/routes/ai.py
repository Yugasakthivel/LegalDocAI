from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from backend.app.core.config import OPENAI_MODEL, UPLOAD_FOLDER
from backend.app.core.deps import get_openai_client, get_async_openai_client, get_current_user, update_openai_key, is_openai_ready
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service, verification_service
from backend.app.services.risk_service import compute_combined_risk_index
from backend.app.services.ml_service import classifier
from backend.app import vectorstore
import time
import random
import re
import os
import shutil
import uuid
import asyncio
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
            {"name": "translate_analyze", "params": ["file", "target_lang", "ocr_lang", "engine"]},
            {"name": "document_types", "params": []}
        ]
    }

@router.get("/document-types")
async def document_types(user: dict = Depends(get_current_user)):
    return {
        "count": len(classifier.get_supported_document_types()),
        "document_types": classifier.get_supported_document_types()
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
                msg = str(e).lower()
                if sc == 429 or "429" in msg or "insufficient_quota" in msg or "exceeded your current quota" in msg:
                    from backend.app.core.deps import mark_quota_exhausted
                    mark_quota_exhausted()
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
        
    retrieved = vectorstore.search_doc_first(question, doc_id, top_k=3)
    contexts = []
    for r in (retrieved or []):
        d_res = collection.find_one({"doc_id": r.get("doc_id")}, {"_id": 0, "combined_text": 1})
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
    return {"answer": answer, "sources": retrieved or []}

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
    if fname == "document_types":
        return await document_types(user)
    raise HTTPException(status_code=400, detail="Unknown function name")

 

@router.post("/translate-analyze")
async def translate_analyze(
    file: UploadFile = File(...),
    target_lang: str = Form("English"),
    ocr_lang: str = Form(None),
    engine: str = Form("auto"),
    user: dict = Depends(get_current_user)
):
    path = None
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file selected.")
        path = os.path.join(UPLOAD_FOLDER, f"trans_{uuid.uuid4()}_{file.filename}")
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        ext = ocr_service.detect_file_type(file.filename)
        warnings = []
        eff_lang = await asyncio.to_thread(ocr_service.resolve_ocr_lang_for_file, path, ext, ocr_lang or "auto")
        if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
            warnings.append("Tamil OCR not installed; using English OCR.")
            eff_lang = "eng"
        pages = await asyncio.to_thread(ocr_service.extract_text_by_filetype, path, ext, ocr_lang=eff_lang)
        src_text = "\n".join(pages)
        if not src_text.strip():
            warnings.append("OCR could not extract any text from the document.")

        engine = (engine or "auto").lower()
        force_openai = engine == "openai"
        force_local = engine == "local"
        if not src_text.strip():
            translated = ""
        elif force_local or (not force_openai and (_prefer_local_translation or not is_openai_ready())):
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
            fb = await asyncio.to_thread(_translate_marian, src_text, target_lang, eff_lang)
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
            client = get_async_openai_client()
            while True:
                try:
                    resp = await client.chat.completions.create(
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
                            await asyncio.sleep(ra + random.uniform(0.05, 0.25))
                        else:
                            d = min(0.5 * (2 ** attempt), 8.0)
                            j = d * 0.3
                            await asyncio.sleep(d + random.uniform(-j, j))
                        attempt += 1
                        if time.perf_counter() - start > 30:
                            raise TimeoutError("Translation timed out")
                        continue
                    raise
    except Exception as e:
        # Fallback logic if the primary translation (OpenAI or Local) fails
        def _detect_script_fallback(text: str) -> str:
            for ch in text:
                o = ord(ch)
                if 0x0B80 <= o <= 0x0BFF:
                    return "tam"
                if 0x0900 <= o <= 0x097F:
                    return "hin"
            return "eng"
        def _marian_model_for_fallback(target: str, source_hint: str) -> Optional[str]:
            t = (target or "").lower()
            s = (source_hint or "").lower()
            if t.startswith("english"):
                if "tam" in s:
                    return "Helsinki-NLP/opus-mt-ta-en"
                if "hin" in s:
                    return "Helsinki-NLP/opus-mt-hi-en"
            return None
        def _translate_marian_fallback(text: str, target: str, source_hint: str) -> Optional[str]:
            model_name = _marian_model_for_fallback(target, source_hint)
            if not model_name:
                auto = _detect_script_fallback(text)
                model_name = _marian_model_for_fallback(target, auto)
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
        
        # We need src_text here, so ensure it's defined or empty
        st = src_text if 'src_text' in locals() else ""
        fb = None
        try:
            fb = await asyncio.to_thread(_translate_marian_fallback, st, target_lang, eff_lang if 'eff_lang' in locals() else "auto")
        except Exception:
            fb = None
        translated = fb or st
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
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    
    async def run_ai_analysis():
        rs, an, working_text = await analysis_service.analyze_text_overall_async(translated)
        base_text = working_text or translated

        tasks = {
            "timeline": analysis_service.build_timeline_async(base_text),
            "doc_type": analysis_service.classify_document_type_async(base_text),
            "clause_risk": analysis_service.clause_risk_detection_async(base_text, translated_text=working_text),
            "signers": analysis_service.validate_signers_async(base_text, rs.get("signers", [])),
            "juris_checks": analysis_service.jurisdiction_specific_checks_async(base_text),
            "juris_curated": analysis_service.curated_jurisdiction_templates_async(base_text),
            "registration": analysis_service.analyze_registration_async(base_text),
            "mandatory": analysis_service.check_mandatory_fields_async(base_text),
            "verification": analysis_service.ask_openai_for_verification_and_confidence_async(base_text),
            "ml_verification": verification_service.verify_document_async(base_text)
        }

        task_keys = list(tasks.keys())
        res_list = await asyncio.gather(*[tasks[k] for k in task_keys], return_exceptions=True)
        res_map = {}
        pipeline_errors = []
        for k, v in zip(task_keys, res_list):
            if isinstance(v, Exception):
                pipeline_errors.append({"stage": k, "error": str(v)})
                # Safe fallbacks per stage
                if k == "timeline":
                    res_map[k] = []
                elif k == "doc_type":
                    res_map[k] = "Generic Legal Document"
                elif k == "clause_risk":
                    res_map[k] = {"clauses": [], "overall_risk": "Unknown", "overall_risk_score": 50}
                elif k == "signers":
                    res_map[k] = {"signers": [], "overall_signer_score": None}
                elif k in {"juris_checks", "juris_curated"}:
                    res_map[k] = {"jurisdiction": None, "checks": [], "overall_compliance_score": 0}
                elif k == "registration":
                    res_map[k] = {"missing_fields": []}
                elif k == "mandatory":
                    res_map[k] = {"missing": [], "present": []}
                elif k == "verification":
                    res_map[k] = {"marker": "UNVERIFIED", "ai_confidence": 30}
                elif k == "ml_verification":
                    res_map[k] = {"marker": "UNVERIFIED", "ai_confidence": 30, "predicted_category": "Generic Legal Document"}
                else:
                    res_map[k] = {}
            else:
                res_map[k] = v

        verification_data = res_map["verification"] or {}
        ml_verification = res_map["ml_verification"] or {}

        effective_marker = ml_verification.get("marker") or verification_data.get("marker")
        effective_ai_confidence = ml_verification.get("ai_confidence")
        if effective_ai_confidence is None:
            effective_ai_confidence = verification_data.get("ai_confidence")
        if effective_ai_confidence is None:
            effective_ai_confidence = verification_data.get("confidence")

        # Keep document type robust: prefer explicit classifier, fallback to ML verification category
        doc_type_primary = res_map["doc_type"]
        doc_type_fallback = ml_verification.get("predicted_category")
        if not doc_type_primary or str(doc_type_primary).strip().lower() in {"generic legal document", "unknown", "other document"}:
            if doc_type_fallback and str(doc_type_fallback).strip().lower() not in {"generic legal document", "unknown", "other document"}:
                doc_type_primary = doc_type_fallback

        an.update({
            "timeline": res_map["timeline"],
            "document_type": doc_type_primary,
            "clause_risk": res_map["clause_risk"],
            "signer_validation": res_map["signers"],
            "jurisdiction_checks": res_map["juris_checks"],
            "jurisdiction_curated": res_map["juris_curated"],
            "registration_analysis": res_map["registration"],
            "mandatory_fields": res_map["mandatory"],
            "verification": verification_data,
            "ml_verification": ml_verification,
            "verified_marker": effective_marker,
            "ai_confidence": effective_ai_confidence
        })

        # Ensure ai_confidence is not 0 if verified
        marker_norm = str(an.get("verified_marker") or "").strip().lower()
        if marker_norm == "verified" and not an.get("ai_confidence"):
            an["ai_confidence"] = 85

        # Ensure bounded confidence value
        try:
            if an.get("ai_confidence") is not None:
                an["ai_confidence"] = int(max(0, min(100, float(an.get("ai_confidence")))))
        except Exception:
            an["ai_confidence"] = 0

        # Scores
        an["legality_score"] = analysis_service.compute_legality_score(an)
        an["combined_risk_index"] = compute_combined_risk_index(an)
        ai_detection_score = analysis_service.compute_ai_detection_score(base_text)
        an["ai_detection_score"] = 0 if ai_detection_score is None else int(max(0, min(100, ai_detection_score)))
        an["genuineness_score"] = analysis_service.compute_genuineness_score(an, ml_verification)
        legal_status = analysis_service.derive_legal_status(an, ml_verification)
        an["legal_status"] = legal_status.get("status")
        an["legal_status_reason"] = legal_status.get("reason")
        an["status_summary"] = legal_status.get("summary")
        an["supported_document_types_count"] = len(classifier.get_supported_document_types())
        an["legal_explanation"] = analysis_service.legal_explanation_engine(base_text)

        required_output = analysis_service.build_required_output_schema(rs, an, base_text)
        an["required_output"] = required_output
        if warnings:
            an["warnings"] = warnings
        if pipeline_errors:
            an["pipeline_errors"] = pipeline_errors
        return {
            "translated_text": base_text,
            "results": rs,
            "analytics": an,
            "output": required_output
        }

    return await run_ai_analysis()

@router.post("/start-analysis")
async def start_analysis(req: StartRequest):
    doc_id = (req.uploadedDocumentId or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="uploadedDocumentId required")
    doc = collection.find_one({"doc_id": doc_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"docId": doc_id, "documentId": doc_id, "uploadedDocumentId": doc_id}
