# backend/app/vectorstore.py

from typing import List, Dict

# Simple in-memory "vector store" example
VECTOR_STORE = [
    {"doc_id": 1, "text": "Legal document about contracts."},
    {"doc_id": 2, "text": "Court case summary and details."},
    {"doc_id": 3, "text": "Intellectual property law document."},
    {"doc_id": 4, "text": "Employment law regulations."},
]

def search_in_index(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for documents containing the query string.
    Returns top_k results.
    """
    if not query:
        return []
    
    # Simple text match scoring
    results = []
    for doc in VECTOR_STORE:
        if query.lower() in doc["text"].lower():
            results.append({"doc_id": doc["doc_id"], "text": doc["text"]})
    
    return results[:top_k]
