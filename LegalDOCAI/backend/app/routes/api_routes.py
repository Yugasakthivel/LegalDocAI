# backend/app/api_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.processing import process_document, process_documents_bulk
from backend.app.vectorstore import search_in_index

# Router object (prefix optional if you want /api/* style)
router = APIRouter()

# -----------------------------
# Request Models
# -----------------------------
class Document(BaseModel):
    text: str

class DocumentsBulk(BaseModel):
    texts: list[str]

class Query(BaseModel):
    query: str
    top_k: int = 5

# -----------------------------
# Routes
# -----------------------------
@router.post("/document")
def add_document(doc: Document):
    """Process and store a single document"""
    try:
        return process_document(doc.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/documents_bulk")
def add_documents_bulk(docs: DocumentsBulk):
    """Process multiple documents in bulk"""
    return process_documents_bulk(docs.texts)

@router.post("/search")
def search(query: Query):
    """Search for relevant documents using FAISS index"""
    results = search_in_index(query.query, query.top_k)
    return {"query": query.query, "results": results}
