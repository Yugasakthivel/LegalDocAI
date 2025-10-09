import io
from fastapi import APIRouter, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from backend.app import processing, vectorstore

router = APIRouter()
@router.post("/upload/")
async def process_file(file: UploadFile = File(...)):
    results = []

    # Read uploaded PDF
    content = await file.read()
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text from PDF page
            text = page.extract_text() or ""

            # If page has no text, do OCR
            if not text.strip():
                # Convert PDF page to image
                img = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(img)

            # Example entity extraction (dummy, you can replace with NLP later)
            entities = []
            if "date" in text.lower():
                entities.append("Date")
            if "name" in text.lower():
                entities.append("Name")
            if "contract" in text.lower():
                entities.append("Contract")

            results.append({
                "page": i + 1,
                "text": text.strip(),
                "entities": entities
            })

    return {"results": results}