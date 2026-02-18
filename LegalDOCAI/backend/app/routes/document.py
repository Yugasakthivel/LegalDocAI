from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from backend.app.core.config import UPLOAD_FOLDER, OPENAI_MODEL
from backend.app.core.deps import get_openai_client, get_current_user
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service
from collections import Counter
import os
import shutil
import uuid
import json
import difflib

router = APIRouter(tags=["document"])

@router.get("/history")
def get_history(user: dict = Depends(get_current_user)):
    """Get all stored document analysis history"""
    try:
        docs = list(collection.find({},{"_id":0}))
        return {"history":docs}
    except Exception as e:
        return {"error":str(e),"history":[]}

@router.get("/document/history")
def get_document_history(user: dict = Depends(get_current_user)):
    """Get all stored document analysis history (alias for /history)"""
    try:
        docs = list(collection.find({},{"_id":0}))
        return {"history":docs}
    except Exception as e:
        return {"error":str(e),"history":[]}

@router.get("/analytics-summary")
def analytics_summary(user: dict = Depends(get_current_user)):
    """Summarize all analytics across documents"""
    try:
        docs = list(collection.find({},{"_id":0}))
        summary = {
            "total_documents": len(docs),
            "total_pages":0,
            "total_names":0,
            "total_emails":0,
            "total_phones":0,
            "total_signers":0,
            "total_clauses":0,
            "legality_scores":[],
            "file_types":{},
            "clause_frequency":Counter()
        }
        for doc in docs:
            analytics = doc.get("analytics",{})
            summary["total_pages"] += analytics.get("total_pages",0)
            summary["total_names"] += analytics.get("total_names",0)
            summary["total_emails"] += analytics.get("total_emails",0)
            summary["total_phones"] += analytics.get("total_phones",0)
            summary["total_signers"] += analytics.get("total_signers",0)
            summary["total_clauses"] += analytics.get("total_clauses",0)
            summary["legality_scores"].append(analytics.get("legality_score",0))
            ftype = analytics.get("file_type","unknown")
            summary["file_types"][ftype] = summary["file_types"].get(ftype,0)+1
            summary["clause_frequency"].update(analytics.get("clause_summary",{}))
        summary["average_legality_score"] = round(sum(summary["legality_scores"])/len(summary["legality_scores"]),2) if summary["legality_scores"] else 0
        summary["clause_frequency"] = dict(summary["clause_frequency"])
        return summary
    except Exception as e:
        return {"error":str(e)}

@router.get("/tech-stack")
async def get_tech_stack():
    return {
        "final_stack": {
            "Backend": "Python",
            "API": "FastAPI",
            "AI_Models": ["GPT", "LLaMA", "Legal-BERT"],
            "OCR": ["Tesseract", "EasyOCR"],
            "Vector_DB": "ChromaDB",
            "Database": "MongoDB",
            "Frontend": "React (Tailwind UI)",
            "Deployment": ["Docker", "AWS/GCP"]
        },
        "rationale": {
            "Python": "Large ecosystem for AI and data processing",
            "FastAPI": "Async, performant, easy schema with Pydantic",
            "GPT/LLaMA/Legal-BERT": "General + open + domain models balance",
            "Tesseract/EasyOCR": "Mature OCR + multilingual fallback",
            "ChromaDB": "Simple local vector store for RAG",
            "MongoDB": "Flexible documents and analytics storage",
            "React (Tailwind UI)": "Modern component UX and rapid styling",
            "Docker/AWS/GCP": "Reproducible deploys and scalable infrastructure"
        },
        "open_source_vs_paid": {
            "Python": "Open-source",
            "FastAPI": "Open-source",
            "GPT": "Paid",
            "LLaMA": "Open-source",
            "Legal-BERT": "Open-source",
            "Tesseract": "Open-source",
            "EasyOCR": "Open-source",
            "ChromaDB": "Open-source",
            "MongoDB": "Open-source",
            "React": "Open-source",
            "Tailwind": "Open-source",
            "Docker": "Open-source",
            "AWS/GCP": "Paid"
        },
        "minimum_system_requirements": {
            "CPU": "4-core",
            "RAM": "8GB",
            "Storage": "50GB",
            "OS": "Windows/Linux/macOS",
            "Dependencies": [
                "Python 3.8+",
                "FastAPI 0.95+",
                "SpaCy 3.5+",
                "MongoDB 5.0+",
                "Tesseract 5.0+",
                "EasyOCR 1.6+",
                "ChromaDB 0.4+"
            ]
        }
    }

@router.post("/compare")
async def compare_documents(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    doc_ids: str = Form(None),
    ocr_lang: str = Form(None),
    user: dict = Depends(get_current_user)
):
    t1 = ""
    t2 = ""

    # 1. Try to load from doc_ids
    if doc_ids:
        try:
            ids = json.loads(doc_ids)
            if isinstance(ids, list) and len(ids) >= 2:
                d1 = collection.find_one({"doc_id": ids[0]}, {"_id": 0, "combined_text": 1})
                d2 = collection.find_one({"doc_id": ids[1]}, {"_id": 0, "combined_text": 1})
                t1 = d1.get("combined_text", "") if d1 else ""
                t2 = d2.get("combined_text", "") if d2 else ""
        except Exception:
            pass

    # 2. Fallback to file uploads if text not ready
    if not t1 or not t2:
        if not file1 or not file2 or not file1.filename or not file2.filename:
            # Only raise if we truly have no text to compare
            if not t1 or not t2:
                raise HTTPException(status_code=400, detail="Provide two files or valid doc_ids.")

        # Extract file 1
        if file1 and file1.filename:
            path1 = os.path.join(UPLOAD_FOLDER, f"cmp_{uuid.uuid4()}_{file1.filename}")
            with open(path1, "wb") as f1:
                shutil.copyfileobj(file1.file, f1)
            ext1 = ocr_service.detect_file_type(file1.filename)
            eff1 = ocr_service.resolve_ocr_lang_for_file(path1, ext1, ocr_lang or "auto")
            if eff1 == "tam" and not ocr_service.is_tesseract_language_available("tam"):
                eff1 = "eng"
            t1_pages = ocr_service.extract_text_by_filetype(path1, ext1, ocr_lang=eff1)
            t1 = "\n".join(t1_pages)
            try: os.remove(path1)
            except: pass

        # Extract file 2
        if file2 and file2.filename:
            path2 = os.path.join(UPLOAD_FOLDER, f"cmp_{uuid.uuid4()}_{file2.filename}")
            with open(path2, "wb") as f2:
                shutil.copyfileobj(file2.file, f2)
            ext2 = ocr_service.detect_file_type(file2.filename)
            eff2 = ocr_service.resolve_ocr_lang_for_file(path2, ext2, ocr_lang or "auto")
            if eff2 == "tam" and not ocr_service.is_tesseract_language_available("tam"):
                eff2 = "eng"
            t2_pages = ocr_service.extract_text_by_filetype(path2, ext2, ocr_lang=eff2)
            t2 = "\n".join(t2_pages)
            try: os.remove(path2)
            except: pass
    
    d = difflib.unified_diff(t1.splitlines(), t2.splitlines(), lineterm="")
    diff_lines = list(d)
    changes_added = [l[1:].strip() for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
    changes_removed = [l[1:].strip() for l in diff_lines if l.startswith("-") and not l.startswith("---")]
    changes_changed = [l.strip() for l in diff_lines if l.startswith("@@")]
    try:
        prompt = (
            "MULTI-DOCUMENT COMPARISON (Optional Advanced)\n"
            "TRAE Node: Comparison Node\n\n"
            "Prompt\n\n"
            "Compare two legal land documents.\n"
            "Identify:\n"
            "- Matching fields\n"
            "- Conflicts\n"
            "- Ownership mismatch\n"
            "Provide summary.\n\n"
            "Return strict JSON with keys:\n"
            "\"matching_fields\":[\"...\"],"
            "\"conflicts\":[\"...\"],"
            "\"ownership_mismatch\":[\"...\"],"
            "\"summary\":\"...\".\n\n"
            "Original document excerpt:\n" + t1[:3000] + "\n\nNew document excerpt:\n" + t2[:3000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=400
        )
        raw = resp.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        cmp_json = json.loads(raw[start:end]) if start != -1 and end > start else {}
    except Exception:
        cmp_json = {}
    matching = cmp_json.get("matching_fields") or []
    conflicts = cmp_json.get("conflicts") or []
    owner_mis = cmp_json.get("ownership_mismatch") or []
    if isinstance(matching, str): matching = [matching]
    if isinstance(conflicts, str): conflicts = [conflicts]
    if isinstance(owner_mis, str): owner_mis = [owner_mis]
    return {
        "added": changes_added[:50],
        "removed": changes_removed[:50],
        "changed": changes_changed[:50],
        "overview": {
            "matching_fields": matching,
            "conflicts": conflicts,
            "ownership_mismatch": owner_mis,
            "summary": cmp_json.get("summary", "")
        }
    }

@router.post("/check-compliance")
async def check_compliance(
    law: str = Form(...),
    text: str = Form(None),
    doc_id: str = Form(None),
    file: UploadFile = File(None),
    ocr_lang: str = Form(None),
    user: dict = Depends(get_current_user)
):
    if not law:
        raise HTTPException(status_code=400, detail="law is required.")
    
    corpus = text or ""
    
    if file:
        try:
            # Save and extract
            ext = ocr_service.detect_file_type(file.filename)
            tmp_path = os.path.join(UPLOAD_FOLDER, f"comp_{uuid.uuid4()}_{file.filename}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            eff_lang = ocr_service.resolve_ocr_lang_for_file(tmp_path, ext, ocr_lang or "auto")
            warning = None
            if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
                warning = "Tamil OCR not installed; using English OCR."
                eff_lang = "eng"
            pages = ocr_service.extract_text_by_filetype(tmp_path, ext, ocr_lang=eff_lang)
            corpus = "\n".join(pages)
            if os.path.exists(tmp_path):
                 os.remove(tmp_path)
        except Exception as e:
            warning = f"OCR extraction failed: {e}"
            corpus = ""

    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text","") if d else ""
        
    if not corpus:
        data = {
            "law": law,
            "compliant": False,
            "violations": [],
            "recommendations": [],
            "confidence": 0,
            "error": "No text provided. Please upload a file, provide doc_id, or paste text."
        }
        if 'warning' in locals() and warning:
            data["warning"] = warning
        return data
        
    try:
        prompt = (
            "Check the document against the specified regulation. "
            "Return strict JSON: {\"law\":\"...\",\"compliant\":true/false,"
            "\"violations\":[{\"rule\":\"...\",\"detail\":\"...\",\"highlight\":\"...\"}],"
            "\"recommendations\":[\"...\"],\"confidence\":0-100}. "
            "Law:\n" + law + "\n\nDocument:\n" + corpus[:6000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=600,
            timeout=20
        )
        raw = resp.choices[0].message.content or ""
        s = raw.find("{"); e = raw.rfind("}") + 1
        data = json.loads(raw[s:e]) if s != -1 and e > s else {}
    except Exception as e:
        msg = str(e)
        err_text = "AI service temporarily unavailable (Quota/Error)." if "quota" in msg.lower() or "OPENAI_QUOTA_EXHAUSTED" in msg else f"Compliance check failed: {e}"
        data = {
            "law": law,
            "compliant": False,
            "violations": [],
            "recommendations": [],
            "confidence": 0,
            "error": err_text
        }
    if 'warning' in locals() and warning:
        try:
            if isinstance(data, dict):
                data["warning"] = warning
        except Exception:
            pass
    return data

@router.post("/check-compliance/upload")
async def check_compliance_upload(
    law: str = Form(...),
    file: UploadFile = File(...),
    ocr_lang: str = Form(None)
):
    return await check_compliance(law=law, text=None, doc_id=None, file=file, ocr_lang=ocr_lang)

@router.post("/verify-document")
async def verify_document(
    text: str = Form(None),
    doc_id: str = Form(None),
    file: UploadFile = File(None),
    ocr_lang: str = Form(None)
):
    corpus = text or ""
    
    if file:
        try:
            # Save and extract
            ext = ocr_service.detect_file_type(file.filename)
            tmp_path = os.path.join(UPLOAD_FOLDER, f"ver_{uuid.uuid4()}_{file.filename}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            eff_lang = ocr_service.resolve_ocr_lang_for_file(tmp_path, ext, ocr_lang or "auto")
            warning = None
            if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
                warning = "Tamil OCR not installed; using English OCR."
                eff_lang = "eng"
            pages = ocr_service.extract_text_by_filetype(tmp_path, ext, ocr_lang=eff_lang)
            corpus = "\n".join(pages)
            if os.path.exists(tmp_path):
                 os.remove(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from uploaded file: {e}")

    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text","") if d else ""
        
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided. Please upload a file, provide doc_id, or paste text.")
    
    v = analysis_service.ask_openai_for_verification_and_confidence(corpus)
    
    if doc_id:
        try:
            collection.update_one(
                {"doc_id": doc_id}, 
                {"$set": {"analytics.verified_marker": v.get("marker"), "analytics.ai_confidence": v.get("ai_confidence")}}
            )
        except Exception:
            pass
    return v

@router.post("/verify-document/upload")
async def verify_document_upload(
    file: UploadFile = File(...),
    ocr_lang: str = Form(None)
):
    return await verify_document(text=None, doc_id=None, file=file, ocr_lang=ocr_lang)

@router.post("/land/verify")
async def land_verify(
    survey_no: str = Form(...),
    district: str = Form(...),
    taluk: str = Form(...),
    village: str = Form(...),
    sro: str = Form(None),
    ec_from: str = Form(None),
    ec_to: str = Form(None),
    doc_id: str = Form(None),
):
    survey_no = (survey_no or "").strip()
    district = (district or "").strip()
    taluk = (taluk or "").strip()
    village = (village or "").strip()
    sro = (sro or "").strip()
    ec_from = (ec_from or "").strip()
    ec_to = (ec_to or "").strip()

    if not survey_no or not district or not taluk or not village:
        raise HTTPException(status_code=400, detail="survey_no, district, taluk, and village are required")

    corpus = ""
    if doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = (d or {}).get("combined_text", "") or ""

    corpus_lc = corpus.lower()
    survey_present_in_doc = bool(corpus and re.search(rf"\b{re.escape(survey_no.lower())}\b", corpus_lc))
    land_classification = analysis_service.extract_land_classification(corpus) if corpus else {
        "primary_land_type": "Unknown",
        "land_types_detected": [],
        "is_property_related": False,
    }
    doc_type = analysis_service.classify_document_type(corpus) if corpus else "Unknown"

    return {
        "request": {
            "survey_no": survey_no,
            "district": district,
            "taluk": taluk,
            "village": village,
            "sro": sro or None,
            "ec_from": ec_from or None,
            "ec_to": ec_to or None,
            "doc_id": doc_id or None,
        },
        "ownership": {
            "status": "matched" if survey_present_in_doc else "not_verified",
            "survey_no_found_in_document": survey_present_in_doc,
            "document_type": doc_type,
            "land_classification": land_classification,
        },
        "ec": {
            "status": "pending_external_registry_check",
            "period": {
                "from": ec_from or None,
                "to": ec_to or None,
            },
        },
        "fmb": {
            "status": "not_connected",
            "note": "FMB verification requires external survey records integration.",
        },
        "zone_checks": {
            "status": "not_connected",
            "note": "Zoning check requires DTCP/CMDA or local planning API integration.",
        },
        "geo": {
            "status": "not_connected",
            "district": district,
            "taluk": taluk,
            "village": village,
        },
    }

@router.post("/classify/document-type")
async def classify_document_type_label(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text", "") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    label = analysis_service.classify_document_type(corpus)
    if doc_id:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.document_type": label}})
        except Exception:
            pass
    return {"label": label}

@router.post("/classify/document-type/upload")
async def classify_document_type_upload(
    file: UploadFile = File(...),
    ocr_lang: str = Form(None)
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        ext = ocr_service.detect_file_type(file.filename)
        tmp_path = os.path.join(UPLOAD_FOLDER, f"cls_{uuid.uuid4()}_{file.filename}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        eff_lang = ocr_service.resolve_ocr_lang_for_file(tmp_path, ext, ocr_lang or "auto")
        if eff_lang == "tam" and not ocr_service.is_tesseract_language_available("tam"):
            eff_lang = "eng"
        pages = ocr_service.extract_text_by_filetype(tmp_path, ext, ocr_lang=eff_lang)
        corpus = "\n".join(pages)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from uploaded file: {e}")

    if not corpus.strip():
        raise HTTPException(status_code=400, detail="OCR returned empty text")

    label = analysis_service.classify_document_type(corpus)
    return {"label": label}
