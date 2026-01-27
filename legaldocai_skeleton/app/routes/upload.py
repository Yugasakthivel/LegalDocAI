import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.models.schemas import UploadResponse
from app.utils.validators import validate_file, sanitize_filename
from app.services import extract_text_from_file, analyze_legal_document
from app.services.database_service import save_document_analysis

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and analyze legal document.
    
    Accepts: PDF, JPG, PNG, DOCX, TXT files
    Returns: Structured analysis with document type, entities, clauses, risk assessment
    
    Current implementation: Skeleton with placeholder OCR and LLM services
    """
    
    file_type = validate_file(file)
    
    sanitized_name = sanitize_filename(file.filename)
    file_path = os.path.join(settings.UPLOAD_FOLDER, sanitized_name)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        extracted_text = extract_text_from_file(file_path, file_type)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
    
    try:
        analysis = analyze_legal_document(extracted_text, document_type=file_type)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to analyze document: {str(e)}")
    
    doc_id = save_document_analysis(
        filename=file.filename,
        file_type=file_type,
        extracted_text=extracted_text,
        analysis=analysis
    )
    
    response = UploadResponse(
        status="success",
        filename=file.filename,
        file_type=file_type,
        extracted_text=extracted_text,
        analysis=analysis,
        disclaimer="This analysis is for educational purposes only and does not constitute legal advice."
    )
    
    return response
