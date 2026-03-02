from typing import List, Dict, Any, Tuple, Optional
import uuid
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.app.database import documents_collection
from backend.app.core.deps import get_openai_client, get_async_openai_client
from backend.app.core.config import EMBEDDING_MODEL
from typing import cast
import asyncio

try:
    import chromadb
except Exception as e:
    # chromadb may fail at import-time if onnxruntime/native deps are missing.
    # Keep backend bootable and fallback to Mongo/local vector logic.
    print(f"[WARN] chromadb unavailable, using fallback vector search only: {e}")
    chromadb = None

# Initialize Chroma (Singleton-ish)
_chroma_client = None
_chroma_collection = None
_fe_pipeline = None
_fe_model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")

def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if not chromadb:
        return None
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", "./chroma"))
            _chroma_collection = _chroma_client.get_or_create_collection(name="legaldoc_docs")
        except Exception:
            _chroma_client = None
            _chroma_collection = None
    return _chroma_collection

def _init_feature_extraction():
    global _fe_pipeline
    if _fe_pipeline is None:
        try:
            from transformers import pipeline
            _fe_pipeline = pipeline("feature-extraction", model=_fe_model_name)
        except Exception:
            _fe_pipeline = None

def _local_embed(texts: List[str]) -> List[List[float]]:
    _init_feature_extraction()
    if _fe_pipeline is None:
        return [[0.0] * 256 for _ in texts]
    try:
        vectors: List[List[float]] = []
        for t in texts:
            feats = _fe_pipeline(t, truncation=True, max_length=512)
            # feats shape: [1, seq_len, hidden_size] or [seq_len, hidden_size]
            if isinstance(feats, list):
                arr = feats[0] if isinstance(feats[0], list) else feats
                if len(arr) == 0:
                    vectors.append([0.0] * 256)
                    continue
                import numpy as np
                vec = np.mean(np.array(arr), axis=0).astype(float).tolist()
                vectors.append(cast(List[float], vec))
            else:
                vectors.append([0.0] * 256)
        return vectors
    except Exception:
        return [[0.0] * 256 for _ in texts]

def local_embed(texts: List[str]) -> List[List[float]]:
    return _local_embed(texts)

def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        resp = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return _local_embed(texts)

async def embed_texts_async(texts: List[str]) -> List[List[float]]:
    try:
        client = get_async_openai_client()
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return await asyncio.to_thread(_local_embed, texts)

def add_document(doc: Dict[str, Any]) -> str:
    """
    Upsert document into MongoDB with computed embedding vector.
    doc expects keys: filename, combined_text (or text), metadata (optional)
    Returns generated doc_id.
    """
    try:
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        text = doc.get("combined_text") or doc.get("text") or ""
        
        # Ensure we have a vector
        vector_list = []
        if text.strip():
            try:
                vectors = embed_texts([text])
                if vectors and len(vectors) > 0:
                    vector_list = vectors[0]
            except Exception as e:
                print(f"[DEBUG] add_document embedding error: {e}")
                vector_list = [0.0] * 256 # Fallback dimension
        
        # Store in Mongo
        db_doc = {
            "doc_id": doc_id,
            "filename": doc.get("filename"),
            "text": text,
            "metadata": doc.get("metadata", {}),
            "vector": vector_list,
        }
        try:
            documents_collection.update_one({"doc_id": doc_id}, {"$set": db_doc}, upsert=True)
        except Exception as e:
            print(f"[DEBUG] add_document mongo error: {e}")
            
        # Store in Chroma if available
        cc = get_chroma_collection()
        if cc and text.strip():
            try:
                cc.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[{"filename": doc.get("filename"), "doc_id": doc_id}]
                )
            except Exception as e:
                print(f"[DEBUG] add_document chroma error: {e}")
        return doc_id
    except Exception as e:
        print(f"[DEBUG] add_document global error: {e}")
        return doc.get("doc_id") or str(uuid.uuid4())

async def add_document_async(doc: Dict[str, Any]) -> str:
    """
    Async version of add_document.
    """
    try:
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        text = doc.get("combined_text") or doc.get("text") or ""
        
        vector_list = []
        if text.strip():
            try:
                vectors = await embed_texts_async([text])
                if vectors and len(vectors) > 0:
                    vector_list = vectors[0]
            except Exception as e:
                print(f"[DEBUG] add_document_async embedding error: {e}")
                vector_list = [0.0] * 256
        
        db_doc = {
            "doc_id": doc_id,
            "filename": doc.get("filename"),
            "text": text,
            "metadata": doc.get("metadata", {}),
            "vector": vector_list
        }
        
        # MongoDB driver update_one is sync but fast, however we can wrap it if we want to be strictly non-blocking
        await asyncio.to_thread(documents_collection.update_one, {"doc_id": doc_id}, {"$set": db_doc}, upsert=True)
        return doc_id
    except Exception as e:
        print(f"[ERROR] vectorstore.add_document_async failed: {e}")
        return ""

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
    try:
        arr = np.array(vecs, dtype=float)
    except Exception:
        arr = np.zeros((0, 0))
    return ids, arr

def search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query string.
    Returns list of {doc_id, score}.
    Prioritizes Chroma if available, else cosine similarity on Mongo vectors.
    """
    if not query: return []
    
    cc = get_chroma_collection()
    if cc:
        try:
            results = cc.query(query_texts=[query], n_results=top_k)
            # Normalize results structure
            out = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                dists = results["distances"][0] if "distances" in results else [0]*len(ids)
                for i, doc_id in enumerate(ids):
                    # Chroma returns distances (lower is better), we want similarity/score.
                    # Just returning raw distance or converting.
                    out.append({"doc_id": doc_id, "score": 1.0 - (dists[i] if i < len(dists) else 0)})
            return out
        except Exception:
            pass # Fallback to manual

    q_vec = embed_texts([query])[0]
    ids, X = get_all_vectors_and_ids()
    if X.size == 0: return []
    
    q_np = np.array([q_vec], dtype=float)
    if q_np.shape[1] != X.shape[1]: return []
    
    sims = cosine_similarity(q_np, X)[0]
    paired = list(zip(ids, sims.tolist()))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [{"doc_id": pid, "score": float(score)} for pid, score in paired[:top_k]]

def search_doc_first(query: str, doc_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not query or not doc_id:
        return []
    cc = get_chroma_collection()
    if cc:
        try:
            results = cc.query(query_texts=[query], n_results=top_k, where={"doc_id": doc_id})
            out = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                dists = results["distances"][0] if "distances" in results else [0]*len(ids)
                for i, did in enumerate(ids):
                    out.append({"doc_id": did, "score": 1.0 - (dists[i] if i < len(dists) else 0)})
            return out
        except Exception:
            pass
    q_vec = embed_texts([query])[0]
    ids, X = get_all_vectors_and_ids()
    if X.size == 0:
        return []
    q_np = np.array([q_vec], dtype=float)
    if q_np.shape[1] != X.shape[1]:
        return []
    sims = cosine_similarity(q_np, X)[0]
    paired = list(zip(ids, sims.tolist()))
    paired = [p for p in paired if p[0] == doc_id]
    paired.sort(key=lambda x: x[1], reverse=True)
    return [{"doc_id": pid, "score": float(score)} for pid, score in paired[:top_k]]

async def search_async(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Async version of search.
    """
    try:
        # Embed query
        query_vectors = await embed_texts_async([query])
        if not query_vectors:
            return []
        query_vec = np.array(query_vectors[0]).reshape(1, -1)
        
        # Get all documents with vectors from Mongo
        ids, all_vectors = await asyncio.to_thread(get_all_vectors_and_ids)
        if not ids:
            return []
            
        # Compute similarities
        similarities = cosine_similarity(query_vec, all_vectors)[0]
        
        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": ids[idx],
                "score": float(similarities[idx])
            })
        return results
    except Exception as e:
        print(f"[ERROR] vectorstore.search_async failed: {e}")
        return []

def fetch_document_by_id(doc_id: str) -> Dict[str, Any]:
    """Return stored document (omit _id)."""
    doc = documents_collection.find_one({"doc_id": doc_id}, {"_id": 0})
    return doc or {}

def chunk_and_index_document(doc_id: str, text: str, chunk_size: int = 300, overlap: int = 50) -> int:
    if not text:
        return 0
    words = text.split()
    if chunk_size <= 0:
        chunk_size = 300
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        end = min(n, i + chunk_size)
        chunk = " ".join(words[i:end])
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i += step
    cc = get_chroma_collection()
    if cc:
        try:
            for idx, chunk in enumerate(chunks):
                cid = f"{doc_id}_chunk_{idx}"
                cc.upsert(ids=[cid], documents=[chunk], metadatas=[{"doc_id": doc_id, "chunk_index": idx}])
        except Exception:
            pass
    return len(chunks)
