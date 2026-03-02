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
from backend.app.services.translation_service import translation_service
from backend.app.nlp import sentence_safe_chunk, detect_language_with_confidence

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
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    def _is_low_confidence_answer(ans: str) -> bool:
        a = (ans or "").strip().lower()
        if not a:
            return True
        weak_signals = [
            "not available in the document",
            "not explicitly stated",
            "not found",
            "document not found",
            "no text content",
        ]
        return any(s in a for s in weak_signals)

    def _ai_fallback_answer() -> str:
        # Last-resort answer path when both doc fetch and RAG retrieval fail.
        try:
            prompt = (
                "No matching document context was found for the user query.\n"
                "Answer in 3-5 lines with clear guidance and explicitly mention that "
                "no document match was found.\n\nUser question:\n" + question
            )
            resp = get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return (
                "No matching document content was found. "
                "Please upload/select the correct document or ask a broader question."
            )

    def _heuristic_legal_answer(q: str, context_blob: str) -> str:
        ql = (q or "").lower()
        ctx = context_blob or ""
        if not ctx.strip():
            return ""
        # Common contract wording: "between <A> and <B>"
        if "parties" in ql or "party" in ql or "owner" in ql:
            m = re.search(
                r"\bbetween\s+([A-Z][A-Za-z\.\s]{1,60}?)\s+and\s+([A-Z][A-Za-z\.\s]{1,60}?)(?:[\,\.\n]|$)",
                ctx,
            )
            if m:
                a = (m.group(1) or "").strip()
                b = (m.group(2) or "").strip()
                if "owner" in ql:
                    return f"Owner appears to be {a}."
                return f"The parties appear to be {a} and {b}."
        if "owner" in ql:
            m = re.search(r"\bowner(?:\s*name)?\s*[:\-]\s*([A-Z][A-Za-z\.\s]{1,80})", ctx, flags=re.IGNORECASE)
            if m:
                return f"Owner appears to be {(m.group(1) or '').strip()}."
        return ""

    def _answer_from_context(context_blob: str) -> str:
        # Fast path first: deterministic methods are both quicker and often
        # more stable for direct extraction questions.
        heuristic = _heuristic_legal_answer(question, context_blob or "")
        if not _is_low_confidence_answer(heuristic):
            return heuristic

        rule_based = analysis_service.answer_with_fallback(question, context_blob or "")
        if not _is_low_confidence_answer(rule_based):
            return rule_based

        # Only call OpenAI when deterministic paths are weak.
        if not is_openai_ready():
            return rule_based
        try:
            prompt = (
                "Answer the question using only the context. "
                "If context does not contain the answer, reply exactly: Not available in the document.\n\n"
                "Context:\n" + (context_blob or "")[:8000] + "\n\nQuestion:\n" + question
            )
            resp = get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return rule_based

    doc = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1, "text": 1, "filename": 1}) if doc_id else None
    doc_found = bool(doc)

    # 1) Primary path: doc-first RAG
    doc_sources: List[Dict[str, Any]] = []
    doc_text = (doc or {}).get("combined_text") or (doc or {}).get("text") or ""
    if doc and doc_text:
        try:
            vectorstore.add_document({"doc_id": doc_id, "filename": doc.get("filename", ""), "combined_text": doc_text})
        except Exception:
            pass
        try:
            doc_sources = vectorstore.search_doc_first(question, doc_id, top_k=3) or []
        except Exception:
            doc_sources = []

        contexts: List[str] = []
        for hit in doc_sources:
            d_res = collection.find_one({"doc_id": hit.get("doc_id")}, {"_id": 0, "combined_text": 1, "text": 1})
            txt = (d_res or {}).get("combined_text") or (d_res or {}).get("text") or ""
            if txt:
                contexts.append(txt[:2000])

        # Fallback to full selected doc text if retrieval is weak
        doc_context = "\n\n".join(contexts) if contexts else (doc_text[:2500])
        if doc_context:
            doc_answer = _answer_from_context(doc_context)
            if not _is_low_confidence_answer(doc_answer):
                return {
                    "answer": doc_answer,
                    "sources": doc_sources,
                    "mode": "rag_doc",
                    "fallback_used": False,
                    "doc_found": doc_found,
                    "message": "Answered from selected document context.",
                }

    # 2) Fallback path: global RAG search across indexed docs
    global_sources: List[Dict[str, Any]] = []
    try:
        global_sources = vectorstore.search(question, top_k=5) or []
    except Exception:
        global_sources = []

    global_contexts: List[str] = []
    for hit in global_sources:
        did = hit.get("doc_id")
        if not did:
            continue
        d_res = vectorstore.fetch_document_by_id(did) or collection.find_one({"doc_id": did}, {"_id": 0, "combined_text": 1, "text": 1})
        txt = (d_res or {}).get("combined_text") or (d_res or {}).get("text") or ""
        if txt:
            global_contexts.append(txt[:2000])
    global_context = "\n\n".join(global_contexts)

    if global_context:
        rag_answer = _answer_from_context(global_context)
        if not _is_low_confidence_answer(rag_answer):
            return {
                "answer": rag_answer,
                "sources": global_sources,
                "mode": "rag_global",
                "fallback_used": True,
                "doc_found": doc_found,
                "message": "Selected document context was unavailable/insufficient, answered via global RAG.",
            }

    # 3) Final fallback: direct AI guidance with explicit no-doc-match notice
    final_answer = _ai_fallback_answer()
    return {
        "answer": final_answer,
        "sources": global_sources or doc_sources or [],
        "mode": "ai_fallback",
        "fallback_used": True,
        "doc_found": doc_found,
        "message": "Document fetch/RAG did not produce a confident answer; used AI fallback.",
    }

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
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        return await chat_rag(doc_id or "", question, user)
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
        elif force_local or (not force_openai and (os.getenv("USE_LOCAL_TRANSLATION_FIRST", "0") == "1" or not is_openai_ready())):
            # Use stateful TranslationService
            chunks = sentence_safe_chunk(src_text, max_chars=1000)
            try:
                # Detect dominant language for the entire document for local translation
                doc_lang, _ = detect_language_with_confidence(src_text)
                translated_chunks = await translation_service.translate_batch(chunks, doc_lang, target_lang)
                translated = " ".join(translated_chunks)
            except Exception as e:
                print(f"[DEBUG] Local translation failed in route: {e}")
                translated = src_text
                warnings.append("Local translation failed; showing original text.")
        else:
            # OpenAI path...
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
        # Final fallback using stateless MyMemory logic from nlp.py or local translation
        st = src_text if 'src_text' in locals() else ""
        try:
            doc_lang, _ = detect_language_with_confidence(st)
            chunks = sentence_safe_chunk(st, max_chars=1000)
            # Try local translation as first fallback
            translated_chunks = await translation_service.translate_batch(chunks, doc_lang, target_lang)
            translated = " ".join(translated_chunks)
        except Exception:
            translated = st
        
        msg = str(e).lower()
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            warnings.append("OpenAI quota exceeded. Used local translation.")
        else:
            warnings.append(f"Translation failed: {str(e)}. Used fallback.")
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
