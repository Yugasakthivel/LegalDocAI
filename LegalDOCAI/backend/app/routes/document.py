from typing import Optional, Dict
from fastapi import APIRouter, Query, HTTPException
from backend.app.vectorstore import fetch_document_by_id
from backend.app.summarizer import summarize_text

router = APIRouter()

# ---------------------- Full document by ID ----------------------
@router.get("/document/{doc_id}")
def get_document(doc_id: str):
    """
    Fetch a full document by ID.
    """
    doc: Optional[Dict] = fetch_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------- Document/Text summary ----------------------
@router.get("/document/summary/")
def get_summary(doc_id: Optional[str] = Query(None), text: Optional[str] = Query(None)):
    """
    Provide either doc_id to summarize stored document,
    or text parameter to summarize raw text.
    """
    if doc_id:
        doc: Optional[Dict] = fetch_document_by_id(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        text_to_summarize: str = doc.get("text") or ""
    elif text:
        text_to_summarize: str = text
    else:
        raise HTTPException(status_code=400, detail="Provide doc_id or text parameter")

    summary: str = summarize_text(text_to_summarize)
    return {"summary": summary}
