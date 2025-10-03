from fastapi import APIRouter, HTTPException, Query
from ..vectorstore import search, fetch_document_by_id

router = APIRouter()

@router.get("/search")
def search_documents(q: str = Query(..., min_length=1), k: int = 5):
    results = search(q, top_k=k)
    # optionally expand to full doc metadata
    expanded = []
    for r in results:
        doc = fetch_document_by_id(r["doc_id"])
        if doc:
            expanded.append({
                "doc_id": r["doc_id"],
                "score": r["score"],
                "filename": doc.get("filename"),
                "text_preview": (doc.get("text") or "")[:400]
            })
    return {"query": q, "results": expanded}
