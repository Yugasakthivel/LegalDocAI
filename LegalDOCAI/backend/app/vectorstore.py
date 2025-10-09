from typing import List, Dict, Any, Tuple
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .database import documents_collection

# NOTE: replace this with your real embedding model (SentenceTransformer) when ready.
# For now we provide a small deterministic placeholder embedding so code runs without heavy installs.
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Simple placeholder embedding: map text -> small numeric vector.
    Replace with SentenceTransformer.encode(...) for realistic embeddings.
    """
    if not texts:
        return []
    vectors = []
    for t in texts:
        # deterministic small vector: [len, sum(chars)%1000, avg_char_code%100]
        length = len(t)
        sumc = sum(ord(c) for c in t) if t else 0
        avg = (sumc // (length + 1)) % 100
        vectors.append([float(length), float(sumc % 1000), float(avg)])
    return vectors


def add_document(doc: Dict[str, Any]) -> str:
    """
    Upsert document into MongoDB with computed embedding vector.
    doc expects keys: filename, combined_text (or text), metadata (optional)
    Returns generated doc_id.
    """
    doc_id = doc.get("doc_id") or str(uuid.uuid4())
    text = doc.get("combined_text") or doc.get("text") or ""
    vector_list = embed_texts([text])[0] if text else []
    db_doc = {
        "doc_id": doc_id,
        "filename": doc.get("filename"),
        "text": text,
        "metadata": doc.get("metadata", {}),
        "vector": vector_list,
    }
    documents_collection.update_one({"doc_id": doc_id}, {"$set": db_doc}, upsert=True)
    return doc_id


def get_all_vectors_and_ids() -> Tuple[List[str], np.ndarray]:
    """
    Retrieve (ids_list, vectors_array) from MongoDB.
    Returns vectors as numpy.ndarray of shape (n_docs, dim). If no vectors, returns empty array shape (0, dim).
    """
    docs = list(documents_collection.find({}, {"doc_id": 1, "vector": 1}))
    ids = [d["doc_id"] for d in docs]
    vecs = [d.get("vector", []) for d in docs]
    if not vecs:
        return ids, np.zeros((0, 0))
    # ensure numeric arrays
    try:
        arr = np.array(vecs, dtype=float)
    except Exception:
        # fallback: empty
        arr = np.zeros((0, 0))
    return ids, arr


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query string.
    Returns list of {doc_id, score}.
    """
    if not query:
        return []

    q_vec = embed_texts([query])[0]
    ids, X = get_all_vectors_and_ids()
    if X.size == 0:
        return []

    q_np = np.array([q_vec], dtype=float)  # shape (1, d)
    # If shapes mismatch, try to pad or return empty
    if q_np.shape[1] != X.shape[1]:
        # shape mismatch: cannot compare
        return []

    sims = cosine_similarity(q_np, X)[0]  # shape (n_docs,)
    paired = list(zip(ids, sims.tolist()))
    paired.sort(key=lambda x: x[1], reverse=True)
    results = [{"doc_id": pid, "score": float(score)} for pid, score in paired[:top_k]]
    return results


def fetch_document_by_id(doc_id: str) -> Dict[str, Any]:
    """Return stored document (omit _id)."""
    doc = documents_collection.find_one({"doc_id": doc_id}, {"_id": 0})
    return doc or {}
