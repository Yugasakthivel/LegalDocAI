from fastapi import APIRouter, UploadFile, File
from backend.app import processing, vectorstore

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a PDF or Image, extracts text, embeds it, and stores vector.
    """
    filename = file.filename or "uploaded_file"
    content = await file.read()

    # Process file content (PDF/Image → text)
    extracted_text = processing.process_uploaded_file_bytes(content, filename)

    # Convert text into embedding
    embedding = vectorstore.embed_text(extracted_text)

    return {
        "filename": filename,
        "extracted_text": extracted_text[:200] + "...",  # preview
        "embedding": embedding.tolist()
    }
