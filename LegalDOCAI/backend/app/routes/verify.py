from fastapi import APIRouter, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncio
from backend.app.services.indian_legal_verifier import IndianLegalVerifier
from backend.app.services.storage_service import get_document, save_document
from backend.app.core.deps import is_openai_ready
from backend.app.core.deps import get_current_user

router = APIRouter(tags=["verify"])
_verifier = IndianLegalVerifier()

@router.get("/status")
async def status():
    return {"openai_ready": is_openai_ready(), "verifier_ready": True}

@router.post("/indian-legal")
async def indian_legal(doc_id: str = Form(None), text: str = Form(None), user: dict = Depends(get_current_user)):
    if not (doc_id or text):
        raise HTTPException(status_code=400, detail="doc_id or text required")
    if doc_id:
        doc = await asyncio.to_thread(get_document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        text = text or (doc.get("combined_text") or doc.get("text") or "")
        if not (text or "").strip():
            raise HTTPException(status_code=400, detail="No text available for verification")
        report = await asyncio.to_thread(_verifier.verify, text)
        try:
            save_document(doc_id, {"analytics.verification_report": report})
        except Exception:
            pass
        return JSONResponse(report)
    report = await asyncio.to_thread(_verifier.verify, text or "")
    return JSONResponse(report)
