from typing import List, Dict
from fastapi import APIRouter, Query
from backend.app.vectorstore import search, fetch_document_by_id

router = APIRouter()

@router.get("/search/")
def search_documents(q: str = Query(..., min_length=1), k: int = 5):
    """
    Perform semantic search on documents.
    q = query (required), k = number of top results (default=5).
    """
    hits: List[Dict] = search(q, top_k=k)
    expanded: List[Dict] = []

    for h in hits:
        doc: Dict = fetch_document_by_id(h.get("doc_id") or "")
        expanded.append({
            "doc_id": h.get("doc_id"),
            "score": h.get("score"),
            "filename": doc.get("filename"),
            "text_preview": (doc.get("text") or "")[:400],
        })

    return {"query": q, "results": expanded}
