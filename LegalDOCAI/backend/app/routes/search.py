from typing import List, Dict
from fastapi import APIRouter, Query, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.app.vectorstore import search, fetch_document_by_id, get_chroma_collection, embed_texts
from backend.app.core.config import OPENAI_API_KEY
from backend.app.core.deps import get_current_user
import uuid

router = APIRouter()


def _openai_configured() -> bool:
    return bool(OPENAI_API_KEY) and not str(OPENAI_API_KEY).startswith("sk-your")

@router.get("/search/")
def search_documents(q: str = Query(..., min_length=1), k: int = 5, user: dict = Depends(get_current_user)):
    """
    Perform semantic search on documents.
    q = query (required), k = number of top results (default=5).
    """
    if not _openai_configured():
        return JSONResponse(
            {"query": q, "results": [], "error": "OpenAI not configured"},
            status_code=503,
        )

    hits: List[Dict] = search(q, top_k=k)
    expanded: List[Dict] = []

    for h in hits:
        doc: Dict = fetch_document_by_id(h.get("doc_id") or "")
        expanded.append({
            "doc_id": h.get("doc_id"),
            "score": h.get("score"),
            "filename": doc.get("filename"),
            "text_preview": (doc.get("text") or doc.get("combined_text") or "")[:400],
        })

    return {"query": q, "results": expanded}

@router.post("/rag/upsert")
async def rag_upsert(text: str = Form(None), doc_id: str = Form(None), source: str = Form(None), user: dict = Depends(get_current_user)):
    if not _openai_configured():
        return JSONResponse({"error": "OpenAI not configured"}, status_code=503)
    chroma_collection = get_chroma_collection()
    if chroma_collection is None:
        raise HTTPException(status_code=503, detail="Vector DB unavailable")
    payload_text = text or ""
    if not payload_text and doc_id:
        d = fetch_document_by_id(doc_id)
        payload_text = d.get("combined_text","") if d else ""
    if not payload_text:
        raise HTTPException(status_code=400, detail="No text to upsert")
    chunk_id = str(uuid.uuid4())
    vecs = embed_texts([payload_text])
    meta = {"doc_id": doc_id, "source": source}
    try:
        chroma_collection.add(ids=[chunk_id], documents=[payload_text], metadatas=[meta], embeddings=vecs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {e}")
    return {"id": chunk_id, "doc_id": doc_id, "size": len(payload_text)}

@router.post("/rag")
async def rag_query_endpoint(doc_id: str = Query(...), question: str = Query(...), user: dict = Depends(get_current_user)):
    """Simple RAG endpoint for frontend chat"""
    if not _openai_configured():
        return JSONResponse(
            {"answer": "OpenAI not configured", "error": "OPENAI_NOT_CONFIGURED"},
            status_code=503,
        )
    # 1. Retrieve document text
    doc = fetch_document_by_id(doc_id)
    if not doc:
        return {"answer": "Document not found."}
    
    text = doc.get("combined_text") or doc.get("text") or ""
    if not text:
        return {"answer": "Document has no text content."}

    # 2. Simple 'search' or 'ask' logic
    # Ideally, we chunk and vector search, but for MVP we can send context to LLM if small enough
    # or use the existing vectorstore functions.
    
    # Check if we have a vector index for this doc? 
    # For now, let's use a direct LLM call with context (since we have the text).
    # This is a fallback if vector store isn't fully set up for this doc yet.
    
    from backend.app.services.analysis_service import answer_with_fallback
    
    # Truncate text to fit context if needed (approx 10k chars)
    context = text[:15000] 
    
    ans = answer_with_fallback(question, context)
    return {"answer": ans}
