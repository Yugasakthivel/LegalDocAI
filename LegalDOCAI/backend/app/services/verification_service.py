import json
import time
import asyncio
from typing import Dict, Any
from backend.app.services.ml_service import classifier
from backend.app.services.analysis_service import validate_signers as regex_validate_signers

def verify_document(full_text: str) -> Dict[str, Any]:
    # Combines risk validation and signer validation into a verification result
    
    # Use Local ML for verification
    # We pass "Unknown" as predicted category for now, or we could let the function predict it again.
    # The ml_service.verify_document_ml handles the prediction internally if needed.
    verification_result = classifier.verify_document_ml(full_text)
    
    return verification_result

async def verify_document_async(full_text: str) -> Dict[str, Any]:
    return await asyncio.to_thread(verify_document, full_text)

def validate_signers(full_text: str, signers: list) -> Dict[str, Any]:
    """
    Validate signers using the regex-based implementation in analysis_service.
    """
    return regex_validate_signers(full_text, signers)

async def validate_signers_async(full_text: str, signers: list) -> Dict[str, Any]:
    return await asyncio.to_thread(validate_signers, full_text, signers)
