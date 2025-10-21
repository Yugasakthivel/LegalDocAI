# main.py — LegalDocAI Backend (Full Verbose Build + AI Confidence + Chart Data)
# Author: Sakthi K | Project: LegalDocAI

import os
import re
import shutil
import json
from collections import Counter
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
from bson import ObjectId

# ---------------- OCR, NLP & File Parsing ----------------
import fitz  # PyMuPDF for PDFs
from PIL import Image
import pytesseract
import spacy
import dateparser
import docx
import pandas as pd

# ---------------- OpenAI SDK ----------------
from openai import OpenAI

# ---------------------- CONFIG IMPORT ----------------------
try:
    from config import (
        OPENAI_API_KEY,
        OPENAI_MODEL,
        MONGO_URI,
        DB_NAME,
        TESSERACT_CMD,
        client as OPENAI_CLIENT,
    )
except Exception:
    try:
        from config import OPENAI_API_KEY, OPENAI_MODEL, MONGO_URI, DB_NAME, TESSERACT_CMD
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        raise ImportError("⚠️ config.py not found or missing required variables") from e

# ---------------------- MONGO SETUP ------------------------
client_mongo = MongoClient(MONGO_URI)
db = client_mongo[DB_NAME]
collection = db["documents"]

# ---------------------- OCR + NLP SETUP --------------------
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    print("⚠️ Warning: TESSERACT_CMD not configured. Using system default.")

# Load SpaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("⚠️ SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

# ---------------------- OPENAI SETUP -----------------------
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY not set in config.py or .env")

client = OPENAI_CLIENT  # use the client from config.py

# ---------------------- VERIFY OPENAI KEY ------------------
def verify_openai_key():
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Ping: verify connectivity (short reply)"}],
            max_tokens=5,
            temperature=0
        )
        print("✅ OpenAI API key validated (connectivity OK).")
        return True
    except Exception as e:
        print("❌ OpenAI API key validation failed:", e)
        return False

verify_openai_key()

# ---------------------- CONSTANTS --------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"\+?\d[\d\s-]{7,}\d"
DATE_PATTERN = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
SIGNER_PATTERN = r"(signed by|signature|authorized signatory|attested by)\s*[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"

CLAUSE_KEYWORDS = [
    "termination", "confidentiality", "liability", "warranty",
    "dispute", "governing law", "payment", "obligation",
    "indemnity", "agreement"
]

# ---------------------- FASTAPI APP ------------------------
app = FastAPI(title="⚖️ LegalDocAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# ---------------------- HELPERS ----------------------------
# ===========================================================

def detect_file_type(filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    if ext in ["pdf", "docx", "xls", "xlsx", "png", "jpg", "jpeg"]:
        return ext
    return "unknown"

def extract_text_from_pdf(file_path: str) -> List[str]:
    texts = []
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text("text") or ""
        if not text.strip():
            # Fallback OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(img)
        texts.append(text)
    return texts

def extract_text_from_image(file_path: str) -> List[str]:
    image = Image.open(file_path)
    return [pytesseract.image_to_string(image)]

def extract_text_from_word(file_path: str) -> List[str]:
    doc = docx.Document(file_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def extract_text_from_excel(file_path: str) -> List[str]:
    df = pd.read_excel(file_path)
    return [df.to_string()]

def extract_text_by_filetype(file_path: str, ext: str) -> List[str]:
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path)
    elif ext == "docx":
        return extract_text_from_word(file_path)
    elif ext in ["xls", "xlsx"]:
        return extract_text_from_excel(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# ---------------------- TEXT ANALYSIS ----------------------
def analyze_text_overall(full_text: str):
    doc = nlp(full_text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    emails = re.findall(EMAIL_PATTERN, full_text)
    phones = re.findall(PHONE_PATTERN, full_text)
    dates = [str(dateparser.parse(d).date()) for d in re.findall(DATE_PATTERN, full_text) if dateparser.parse(d)]
    clause_counter = Counter()
    for kw in CLAUSE_KEYWORDS:
        clause_counter[kw] = len(re.findall(r"\b" + re.escape(kw) + r"\b", full_text, flags=re.IGNORECASE))
    signers = [m[1] for m in re.findall(SIGNER_PATTERN, full_text, flags=re.IGNORECASE)]
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    keyword_frequency = dict(Counter(tokens).most_common(10))

    # Summarization
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
    summary = "No summary available."
    if sentences:
        freq = Counter(t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct)
        maxf = max(freq.values()) if freq else 1
        sent_scores = {}
        for s in sentences:
            s_doc = nlp(s.lower())
            score = sum(freq.get(t.lemma_.lower(),0)/maxf for t in s_doc)
            sent_scores[s] = score
        summary = " ".join(s for s,_ in sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:3])

    total_clauses = sum(clause_counter.values())
    legality_score = min(100, int(
        min(40, len(set(names))*2) +
        min(30, total_clauses*4) +
        min(30, (len(set(emails))+len(set(phones)))*2)
    ))

    results_summary = {
        "names": list(set(names)),
        "organizations": list(set(orgs)),
        "emails": list(set(emails)),
        "phones": list(set(phones)),
        "dates": list(set(dates)),
        "clauses_found": list(clause_counter.keys()),
        "signers": list(set(signers))
    }

    analytics = {
        "clause_summary": dict(clause_counter),
        "keyword_frequency": keyword_frequency,
        "summary": summary,
        "legality_score": legality_score,
        "total_names": len(results_summary["names"]),
        "total_emails": len(results_summary["emails"]),
        "total_phones": len(results_summary["phones"]),
        "total_signers": len(results_summary["signers"]),
        "total_clauses": total_clauses
    }

    return results_summary, analytics

# ---------------------- OPENAI VERIFICATION -----------------
def ask_openai_for_verification_and_confidence(text: str) -> Dict[str, Any]:
    try:
        prompt = (
            "You are a legal document verification assistant. "
            "Given the document text, answer ONLY in strict JSON with fields: "
            "\"status\" (LEGAL or UNVERIFIED), \"confidence\" (integer 0-100), and an optional \"note\".\n\n"
            "Document (first 2000 chars):\n" + text[:2000]
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=150
        )

        raw = resp.choices[0].message.content.strip() # type: ignore
        json_str = None
        if raw.startswith("{"):
            json_str = raw
        else:
            start = raw.find("{")
            end = raw.rfind("}")+1
            if start != -1 and end != -1 and end>start:
                json_str = raw[start:end]
        if json_str:
            parsed = json.loads(json_str)
            status = parsed.get("status","UNVERIFIED").upper()
            marker = "✅ LEGAL" if "LEGAL" in status else "❌ UNVERIFIED"
            confidence = int(float(parsed.get("confidence",0)))
            confidence = max(0,min(100,confidence))
            return {"marker": marker, "ai_confidence": confidence, "raw": raw}
        else:
            marker = "✅ LEGAL" if "LEGAL" in raw.upper() else "❌ UNVERIFIED"
            m = re.search(r"(\d{1,3})\s*%", raw)
            confidence = int(m.group(1)) if m else None
            return {"marker": marker, "ai_confidence": confidence, "raw": raw}

    except Exception as e:
        print(f"⚠️ OpenAI verification failed: {e}")
        return {"marker":"❌ UNVERIFIED","ai_confidence":None,"raw":str(e)}

# ===========================================================
# ---------------------- ROUTES -----------------------------
# ===========================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload, scan, analyze, verify and save document"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    # save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    file_type = detect_file_type(file.filename)
    try:
        page_texts = extract_text_by_filetype(file_path,file_type)
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Failed to extract text: {e}")

    if not any(page_texts):
        raise HTTPException(status_code=400,detail="Failed to extract text from file.")

    full_text = "\n".join(page_texts)
    results_summary, analytics = analyze_text_overall(full_text)
    analytics["file_type"] = file_type
    analytics["total_pages"] = len(page_texts)

    # OpenAI verification
    try:
        openai_verif = ask_openai_for_verification_and_confidence(full_text)
        analytics["verified_marker"] = openai_verif.get("marker","❌ UNVERIFIED")
        analytics["ai_confidence"] = openai_verif.get("ai_confidence",None)
        analytics["openai_raw"] = openai_verif.get("raw","")[:2000]
    except Exception as e:
        analytics["verified_marker"]="❌ UNVERIFIED"
        analytics["ai_confidence"]=None
        analytics["openai_raw"]=f"Error:{e}"

    # Page-wise NLP
    results=[]
    for i,txt in enumerate(page_texts):
        pdoc = nlp(txt)
        page_names = [ent.text for ent in pdoc.ents if ent.label_=="PERSON"]
        page_orgs = [ent.text for ent in pdoc.ents if ent.label_=="ORG"]
        page_emails = re.findall(EMAIL_PATTERN, txt)
        page_phones = re.findall(PHONE_PATTERN, txt)
        page_clauses = [kw for kw in CLAUSE_KEYWORDS if re.search(r"\b"+re.escape(kw)+r"\b",txt,flags=re.IGNORECASE)]
        page_signers = [m[1] for m in re.findall(SIGNER_PATTERN, txt, flags=re.IGNORECASE)]
        results.append({
            "page": i+1,
            "names": page_names,
            "organizations": page_orgs,
            "emails": page_emails,
            "phones": page_phones,
            "clauses_found": page_clauses,
            "signers": page_signers,
            "text": txt
        })

    # Chart data
    nlp_score = analytics.get("legality_score",0)
    ai_conf = analytics.get("ai_confidence")
    combined_confidence = ai_conf if isinstance(ai_conf,int) else nlp_score
    chart_data = {
        "labels":["NLP Legality Score","AI Confidence"],
        "values":[nlp_score, ai_conf if ai_conf is not None else nlp_score],
        "combined_confidence": combined_confidence
    }
    analytics["chart_data"] = chart_data

    # MongoDB insert
    try:
        doc_id = str(ObjectId())
        doc_data = {
            "doc_id": doc_id,
            "filename": file.filename,
            "results": results,
            "analytics": analytics
        }
        collection.insert_one(doc_data)
    except Exception as e:
        print(f"⚠️ MongoDB insert failed: {e}")

    return JSONResponse({
        "fileName": file.filename,
        "results": results,
        "analytics": analytics
    })

# ---------------- AI Question Route ----------------
class AIRequest(BaseModel):
    text: str
    question: str = None

@app.post("/api/ai-response")
async def ai_response(req: AIRequest):
    """Ask custom questions or summarize document text"""
    prompt = f"{req.text}\n\nQuestion: {req.question}\nAnswer briefly:" if req.question else f"Summarize the following document text briefly:\n\n{req.text}"
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return {"answer":response.choices[0].message.content}
    except Exception as e:
        return {"answer":f"⚠️ Failed to call OpenAI: {e}"}

# ---------------- History ----------------
@app.get("/history")
def get_history():
    """Get all stored document analysis history"""
    try:
        docs = list(collection.find({},{"_id":0}))
        return {"history":docs}
    except Exception as e:
        return {"error":str(e),"history":[]}

# ---------------- Analytics Summary ----------------
@app.get("/analytics-summary")
def analytics_summary():
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

# ===========================================================
# ---------------------- RUN SERVER -------------------------
# ===========================================================
if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
