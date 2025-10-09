from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app import processing, vectorstore

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, extract text (OCR/PDF/Text), embed, and store in vector DB.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file sent")

    # filename fallback
    filename: str = file.filename or "uploaded_file"
    file_bytes = await file.read()

    # Process file -> returns dict with combined_text and OCR results
    result: dict = processing.process_uploaded_file_bytes(file_bytes, filename)

    # Make sure combined_text is str
    combined_text: str = str(result.get("combined_text") or "")

    # Prepare document for vector store
    doc = {
        "filename": filename,
        "combined_text": combined_text,
        "metadata": {"source_filename": filename},
    }

    # Store document
    doc_id = vectorstore.add_document(doc)

    # Safe count of OCR texts
    ocr_texts = list(result.get("ocr_texts", []))

    return {
        "doc_id": doc_id,
        "filename": filename,
        "text_preview": combined_text[:500],
        "ocr_texts_count": len(ocr_texts),
    }
