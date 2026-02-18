from fastapi import APIRouter, HTTPException, Form
from backend.app.core.deps import get_openai_client
from backend.app.core.config import OPENAI_MODEL
from backend.app.database import documents_collection as collection, db
import json
import datetime
import os

router = APIRouter(tags=["dataset"])

@router.post("/explanation")
async def dataset_explanation(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text", "") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        prompt = (
            "LEGAL EXPLANATION DATASET \n"
            "Explain this legal document in simple English.\n\n"
            "Cover:\n"
            "- What document it is\n"
            "- Purpose\n"
            "- Legal importance\n"
            "- Limitations\n\n"
            "Keep explanation under 120 words.\n\n"
            "OCR Text:\n" + corpus[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220
        )
        explanation = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        try:
            from backend.app.services.analysis_service import summarize_fallback
            explanation = summarize_fallback(corpus)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Dataset explanation failed: {e}")
    if doc_id and explanation:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.explanation_dataset": explanation}})
        except Exception:
            pass
    return {"explanation": explanation}

@router.post("/qa")
async def dataset_qa(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text", "") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        prompt = (
            "LEGAL Q&A DATASET (RAG TRAINING)\n"
            "Generate 5 question-answer pairs from this legal document.\n\n"
            "Rules:\n"
            "- Questions must be answerable only from the document\n"
            "- If data is missing for an answer, use exactly: Not available in document\n\n"
            "Return strict JSON only in the following format:\n"
            "[\n"
            "  {\"question\": \"\", \"answer\": \"\"},\n"
            "  ...\n"
            "]\n\n"
            "Document text:\n" + corpus[:5000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        raw = resp.choices[0].message.content or ""
        start = raw.find("[")
        end = raw.rfind("]") + 1
        data = json.loads(raw[start:end]) if start != -1 and end > start else []
    except Exception as e:
        try:
            from backend.app.services.analysis_service import answer_with_fallback
            # Simple heuristic: generate 5 generic questions and answer via extractive QA
            base_qs = [
                "What is the document type?",
                "Who are the parties mentioned?",
                "What are the key dates?",
                "What obligations or payments are specified?",
                "What is the governing law?"
            ]
            data = []
            for q in base_qs:
                a = answer_with_fallback(q, corpus)
                if not a.strip():
                    a = "Not available in document"
                data.append({"question": q, "answer": a})
        except Exception:
            raise HTTPException(status_code=500, detail=f"Q&A dataset generation failed: {e}")
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("question", "")
        a = item.get("answer", "")
        if q is None: q = ""
        if a is None: a = ""
        if not isinstance(q, str): q = str(q)
        if not isinstance(a, str): a = str(a)
        if not q.strip():
            continue
        if not a.strip():
            a = "Not available in document"
        cleaned.append({"question": q.strip(), "answer": a.strip()})
    if len(cleaned) > 5:
        cleaned = cleaned[:5]
    if doc_id and cleaned:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.qa_dataset": cleaned}})
        except Exception:
            pass
    return {"qa_pairs": cleaned}

@router.post("/scan")
async def dataset_scan(root: str = Form(None), limit: int = Form(None)):
    from backend.app.services.ocr_service import extract_text, detect_file_type
    from backend.app.services.storage_service import create_initial_document
    from backend.app.routes.pipeline import run_full_pipeline
    import uuid

    base = root or os.environ.get("DATASET_ROOT", os.path.join(os.getcwd(), "legal_dataset"))
    raw = os.path.join(base, "raw_documents")
    if not os.path.isdir(raw):
        # Fallback to a folder named 'dataset' in root if 'legal_dataset/raw_documents' doesn't exist
        raw = os.path.join(os.getcwd(), "dataset")
        if not os.path.isdir(raw):
            raise HTTPException(status_code=404, detail="Dataset folder not found. Please create 'legal_dataset/raw_documents' or 'dataset' folder.")
    
    allowed = {"pdf","docx","xls","xlsx","png","jpg","jpeg","txt"}
    processed = []
    skipped = []
    
    # Get list of files
    files = [f for f in os.listdir(raw) if os.path.isfile(os.path.join(raw, f))]
    if limit:
        files = files[:limit]

    for filename in files:
        ext = detect_file_type(filename)
        if ext not in allowed:
            skipped.append({"filename": filename, "reason": f"Unsupported extension: {ext}"})
            continue
            
        file_path = os.path.join(raw, filename)
        try:
            # 1. Extract Text
            text = extract_text(file_path, file_type=ext)
            if not text.strip():
                skipped.append({"filename": filename, "reason": "No text extracted"})
                continue
                
            # 2. Create DB Entry
            doc_id = str(uuid.uuid4())
            create_initial_document(doc_id, filename, ext, text)
            
            # 3. Run Pipeline
            # run_full_pipeline is async, but we can call it here
            await run_full_pipeline(doc_id)
            
            processed.append({"filename": filename, "doc_id": doc_id})
        except Exception as e:
            skipped.append({"filename": filename, "reason": str(e)})

    return {
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "processed": processed,
        "skipped": skipped
    }

@router.post("/synthetic/land-record")
async def synthetic_land_record(
    doc_type: str = Form("Patta"),
    count: int = Form(1)
):
    try:
        dt = (doc_type or "Patta").strip()
        if dt.lower() not in ["patta", "tslr"]:
            dt = "Patta"
        n = count if isinstance(count, int) else 1
        if n < 1: n = 1
        if n > 20: n = 20
        texts = []
        base = (
            "SYNTHETIC LEGAL DOCUMENT GENERATION\n\n"
            "Use this in TRAE / LLM to generate training data.\n\n"
            "Generate a realistic Indian land record document similar to a {doctype}.\n\n"
            "Requirements:\n"
            "- Mix Tamil and English text\n"
            "- Include District, Taluk, Village, Survey Number, Owner Name\n"
            "- Add government-style wording\n"
            "- Include fake reference number and issue date\n"
            "- Make it look like OCR-extracted text (imperfect formatting)\n\n"
            "Output as raw text only. Do not add explanations."
        )
        prompt = base.format(doctype=("Patta" if dt.lower() == "patta" else "Town Survey Land Register"))
        for _ in range(n):
            resp = get_openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=900
            )
            t = (resp.choices[0].message.content or "").strip()
            texts.append(t)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic generation failed: {e}")
    return {"doc_type": dt, "count": n, "texts": texts, "text": texts[0] if texts else ""}

@router.post("/annotations/extract")
async def annotations_extract(text: str = Form(None), doc_id: str = Form(None)):
    corpus = text or ""
    if not corpus and doc_id:
        d = collection.find_one({"doc_id": doc_id}, {"_id": 0, "combined_text": 1})
        corpus = d.get("combined_text", "") if d else ""
    if not corpus:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        prompt = (
            "ANNOTATION GENERATION (GROUND TRUTH)\n"
            "You are a legal data annotator. Given the OCR text of a land document, extract the following fields and return JSON:\n"
            "{ \"Document_Type\": \"\", \"District\": \"\", \"Taluk\": \"\", \"Village\": \"\", \"Survey_Number\": \"\", \"Owner_Name\": \"\", \"Land_Class\": \"\", \"Extent\": \"\", \"Issue_Date\": \"\", \"Reference_Number\": \"\" }\n"
            "Ensure correctness and consistency. If a field is not present, use an empty string. Return only JSON.\n\n"
            "OCR Text:\n" + corpus[:4000]
        )
        resp = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        raw = resp.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end]) if start != -1 and end > start else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annotation generation failed: {e}")
    keys = ["Document_Type","District","Taluk","Village","Survey_Number","Owner_Name","Land_Class","Extent","Issue_Date","Reference_Number"]
    for k in keys:
        v = data.get(k, "")
        if v is None: v = ""
        if not isinstance(v, str): v = str(v)
        data[k] = v
    if doc_id:
        try:
            collection.update_one({"doc_id": doc_id}, {"$set": {"analytics.annotations": data}})
        except Exception:
            pass
    return {"annotations": data} 
