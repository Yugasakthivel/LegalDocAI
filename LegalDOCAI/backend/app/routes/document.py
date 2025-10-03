from fastapi import APIRouter, Query
from ..summarizer import summarize_text
from ..vectorstore import fetch_document_by_id

router = APIRouter()

@router.get("/document/summary")
def get_summary(doc_id: str = Query(None), text: str = Query(None)):
    """
    If doc_id is provided, summarize stored document.
    Else if text provided, summarize that text.
    """
    if doc_id:
        doc = fetch_document_by_id(doc_id)
        if not doc:
            return {"error": "Document not found"}
        text_to_summary = doc.get("text", "")
    elif text:
        text_to_summary = text
    else:
        return {"error": "Provide doc_id or text parameter"}

    summary = summarize_text(text_to_summary)
    return {"summary": summary}
