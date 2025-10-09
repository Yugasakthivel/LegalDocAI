# main.py

import os
import re
import shutil
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import spacy
import dateparser

# ---------------------- NLP Model ----------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------- Patterns ----------------------
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"\+?\d[\d\s-]{7,}\d"
DATE_PATTERN = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

CLAUSE_KEYWORDS = [
    "termination", "confidentiality", "liability", "warranty",
    "dispute", "governing law", "payment", "obligation",
    "indemnity", "agreement"
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- FastAPI App ----------------------
app = FastAPI(title="LegalDocAI Backend")

# ---------------------- CORS ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"] for frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Health Check ----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------- Upload Only (optional) ----------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Uploaded {file.filename} successfully", "filename": file.filename}

# ---------------------- PDF Processing ----------------------
def generate_summary(text: str, max_sentences=5):
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
    if not sentences:
        return "No summary available."

    word_freq = {}
    for word in doc:
        if word.is_stop or word.is_punct:
            continue
        word_freq[word.lemma_.lower()] = word_freq.get(word.lemma_.lower(), 0) + 1

    max_freq = max(word_freq.values(), default=1)
    for w in word_freq:
        word_freq[w] /= max_freq

    sent_scores = {}
    for sent in sentences:
        sent_doc = nlp(sent.lower())
        sent_scores[sent] = sum(word_freq.get(token.lemma_, 0) for token in sent_doc)

    top_sents = [s for s, _ in sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]]
    return " ".join(top_sents)

def calculate_legality_score(results) -> int:
    total_pages = len(results)
    total_clauses = sum(len(r["clauses_found"]) for r in results)
    legal_keywords_found = sum(
        sum(1 for kw in CLAUSE_KEYWORDS if kw.lower() in r["text"].lower())
        for r in results
    )

    score = 0
    if total_pages > 0:
        score += min(50, total_pages * 5)
    score += min(30, total_clauses * 3)
    score += min(20, legal_keywords_found * 2)
    return min(100, score)

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    results = []
    clause_counter = Counter()
    total_pages = 0
    full_text = ""

    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)

        for i, page in enumerate(list(doc)): # type: ignore
            text = page.get_text("text") or ""

            # OCR fallback for scanned pages
            if not text.strip():
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img)

            full_text += text + "\n"

            # Regex extraction
            emails = re.findall(EMAIL_PATTERN, text)
            phones = re.findall(PHONE_PATTERN, text)
            dates = [
                str(dateparser.parse(d))
                for d in re.findall(DATE_PATTERN, text)
                if dateparser.parse(d)
            ]

            # NLP entity recognition
            doc_nlp = nlp(text)
            names = [ent.text for ent in doc_nlp.ents if ent.label_ == "PERSON"]
            orgs = [ent.text for ent in doc_nlp.ents if ent.label_ == "ORG"]

            # Clause detection
            clauses_found = []
            for kw in CLAUSE_KEYWORDS:
                if kw.lower() in text.lower():
                    clauses_found.append(kw)
                    clause_counter[kw] += 1

            results.append({
                "page": i + 1,
                "names": names,
                "emails": emails,
                "phones": phones,
                "organizations": orgs,
                "dates": dates,
                "clauses_found": clauses_found,
                "text": text.strip()
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    summary = generate_summary(full_text, max_sentences=6)
    legality_score = calculate_legality_score(results)

    analytics = {
        "total_pages": total_pages,
        "total_names": sum(len(r["names"]) for r in results),
        "total_emails": sum(len(r["emails"]) for r in results),
        "total_phones": sum(len(r["phones"]) for r in results),
        "total_clauses": sum(len(r["clauses_found"]) for r in results),
        "clause_summary": dict(clause_counter),
        "summary": summary,
        "legality_score": legality_score
    }

    return JSONResponse({"results": results, "analytics": analytics})

# ---------------------- Root ----------------------
@app.get("/")
def root():
    return {"message": "Welcome to LegalDocAI backend (final version)!"}

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
