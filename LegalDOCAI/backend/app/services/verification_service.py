import json
import time
import asyncio
from backend.app.services.analysis_service import validate_signers as analysis_validate_signers
from backend.app.services.ml_service import classifier
from backend.app.services.visual_forensics_service import visual_forensics
from PIL import Image
from typing import Dict, Any, List, Optional

def verify_document(full_text: str, images: Optional[List[Image.Image]] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    # 1. Textual Analysis (Existing)
    text_result = classifier.verify_document_ml(full_text)
    
    # 2. Visual Forensics (New)
    visual_result = None
    
    # If images not provided but file_path is, load first page
    if not images and file_path and file_path.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(file_path)
            if len(doc) > 0:
                pix = doc[0].get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images = [img]
        except Exception:
            pass
    elif not images and file_path and file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            images = [Image.open(file_path)]
        except Exception:
            pass

    if images:
        # Analyze the first page for visual authenticity
        visual_result = visual_forensics.detect_visual_fraud(images[0])
        
    # 3. Score Fusion Logic
    if visual_result and "error" not in visual_result:
        v_score = visual_result["visual_score"]
        t_score = text_result["ai_confidence"]
        
        # Combined score (weighted average)
        # We give more weight to text for legal content, but visual for forgery
        fused_score = (t_score * 0.6) + (v_score * 0.4)
        
        text_result["ai_confidence"] = round(fused_score, 2)
        text_result["visual_forensics"] = visual_result
        
        # Adjust marker based on visual red flags
        if visual_result["verdict"] == "Fake" and text_result["marker"] != "Rejected":
            text_result["marker"] = "Flagged"
            text_result["issues"].append("Visual forensics detected high probability of tampering/forgery.")
            
    return text_result

async def verify_document_async(full_text: str, images: Optional[List[Image.Image]] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(verify_document, full_text, images, file_path)

def validate_signers(full_text: str, signers: list) -> Dict[str, Any]:
    """
    Validate signers using the shared analysis_service implementation.
    """
    return analysis_validate_signers(full_text, signers)

async def validate_signers_async(full_text: str, signers: list) -> Dict[str, Any]:
    return await asyncio.to_thread(validate_signers, full_text, signers)
