import os
import uuid
from fastapi import UploadFile, HTTPException
from app.config import settings

def get_file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def validate_file(file: UploadFile) -> str:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = get_file_extension(file.filename)
    
    if not file_ext:
        raise HTTPException(status_code=400, detail="File must have an extension")
    
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )
    
    return file_ext

def sanitize_filename(original_filename: str) -> str:
    file_ext = get_file_extension(original_filename)
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    return unique_name
