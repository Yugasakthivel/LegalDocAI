import os
import re
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from backend.app.core.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_MIN_CONFIDENCE

try:
    from backend.app.vectorstore import local_embed as _vector_local_embed
except Exception:
    _vector_local_embed = None

try:
    from backend.app.vectorstore import embed_texts as _vector_embed_texts
except Exception:
    _vector_embed_texts = None

def get_rag_mode() -> str:
    m = (os.environ.get("RAG_MODE", "auto") or "auto").lower().strip()
    if m not in {"openai", "local", "auto"}:
        return "auto"
    return m

def chunk_document(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if not text:
        return []
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
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    try:
        if _vector_local_embed:
            return _vector_local_embed(chunks)
        if _vector_embed_texts:
            return _vector_embed_texts(chunks)
    except Exception:
        pass
    return [[0.0] * 256 for _ in chunks]

def retrieve_top_chunks(question: str, chunks: List[str], chunk_vectors: List[List[float]], top_k: int = 3) -> List[Tuple[str, float]]:
    if not question or not chunks or not chunk_vectors:
        return []
    try:
        if _vector_local_embed:
            q_vec = _vector_local_embed([question])[0]
        elif _vector_embed_texts:
            q_vec = _vector_embed_texts([question])[0]
        else:
            q_vec = [0.0] * 256
    except Exception:
        q_vec = [0.0] * 256
    q_np = np.array([q_vec], dtype=float)
    X = np.array(chunk_vectors, dtype=float)
    if q_np.ndim != 2 or X.ndim != 2 or q_np.shape[1] != X.shape[1]:
        sims = [0.0] * len(chunks)
    else:
        sims = cosine_similarity(q_np, X)[0].tolist()
    legal_terms = {"termination", "notice", "payment", "rent", "liability", "indemnity", "penalty", "jurisdiction", "arbitration", "confidential"}
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    boosted: List[Tuple[str, float]] = []
    for i, ch in enumerate(chunks):
        c_words = set(re.findall(r"[a-zA-Z]+", ch.lower()))
        bonus = 0.0
        for t in legal_terms:
            if t in q_words and t in c_words:
                bonus += 0.05
        boosted.append((ch, sims[i] + bonus))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted[:max(1, top_k)]

def extract_answer(question: str, context_chunks: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not context_chunks:
        return "The document may contain this information but it could not be extracted automatically. Try rephrasing your question.", 0.0
    top_chunk = context_chunks[0][0]
    raw_score = float(context_chunks[0][1])
    sents = re.split(r"[.!?]+", top_chunk)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return "The document may contain this information but it could not be extracted automatically. Try rephrasing your question.", 0.0
    q_tokens = set(re.findall(r"[a-zA-Z]+", question.lower()))
    best_sent = sents[0]
    best_score = 0
    for s in sents:
        s_tokens = set(re.findall(r"[a-zA-Z]+", s.lower()))
        score = len(q_tokens.intersection(s_tokens))
        if score > best_score:
            best_score = score
            best_sent = s
    norm = float(best_score) / float(len(q_tokens) or 1)
    if norm < float(RAG_MIN_CONFIDENCE):
        fallback_text = " ".join(sents[:2]) if len(sents) >= 2 else sents[0]
        if raw_score < float(RAG_MIN_CONFIDENCE):
            return "The document may contain this information but it could not be extracted automatically. Try rephrasing your question.", 0.0
        return fallback_text, raw_score
    return best_sent, norm

def _collect_name_candidates(doc_text: str, doc_analytics: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    a = doc_analytics or {}
    nested_entities = (a.get("entities") or {}) if isinstance(a, dict) else {}
    candidates = []
    candidates.extend(a.get("names") or [])
    candidates.extend(nested_entities.get("names") or [])
    for n in candidates:
        s = re.sub(r"\s+", " ", str(n or "")).strip(" .,:;-")
        if not s:
            continue
        if re.search(r"\d", s):
            continue
        if len(s) < 3 or len(s) > 80:
            continue
        if s.lower() not in {x.lower() for x in out}:
            out.append(s)

    if out:
        return out[:8]

    t = doc_text or ""
    pats = [
        r"\b(?:name|holder name|owner name|applicant name)\s*[:\-]\s*([A-Za-z][A-Za-z\s\.\-']{2,80})",
        r"\b(?:signed by|signature of|authorized signatory)\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.\-']{2,80})",
    ]
    for pat in pats:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            s = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" .,:;-")
            if not s:
                continue
            if re.search(r"\d", s):
                continue
            if len(s.split()) < 2:
                continue
            if s.lower() not in {x.lower() for x in out}:
                out.append(s)
            if len(out) >= 8:
                return out
    return out[:8]

def generate_answer_offline(question: str, doc_text: str, doc_analytics: Dict[str, Any]) -> Dict[str, Any]:
    q = (question or "").lower()
    a = doc_analytics or {}
    if any(k in q for k in ["survey", "s.no", "sno", "survey no"]):
        try:
            m = re.search(r"(s\.?no\.?|survey\s*no\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", (doc_text or ""), flags=re.IGNORECASE)
            if m:
                return {"mode": "offline", "answer": f"Survey number: {m.group(2)}", "confidence": 0.9, "source": "regex", "chunks_used": 0}
            meta = ((a or {}).get("metadata") or {})
            sns = meta.get("survey_numbers") or []
            if isinstance(sns, list) and sns:
                return {"mode": "offline", "answer": f"Survey number: {sns[0]}", "confidence": 0.8, "source": "metadata", "chunks_used": 0}
        except Exception:
            pass
    if any(k in q for k in ["name of doc", "document name", "doc name", "file name", "filename"]):
        v = a.get("filename") or a.get("document_name") or (a.get("analytics", {}) or {}).get("filename")
        if v:
            return {"mode": "offline", "answer": str(v), "confidence": 1.0, "source": "local_fallback", "chunks_used": 0}
    if "type" in q or "document type" in q:
        v = a.get("document_type") or a.get("analytics", {}).get("document_type")
        if v:
            return {"mode": "offline", "answer": str(v), "confidence": 1.0, "source": "local_fallback", "chunks_used": 0}
    if "status" in q or "verification" in q:
        v = a.get("verification_marker") or a.get("legal_status") or a.get("analytics", {}).get("legal_status")
        if v:
            return {"mode": "offline", "answer": str(v), "confidence": 1.0, "source": "local_fallback", "chunks_used": 0}
    if "confidence" in q:
        v = a.get("ai_confidence") or a.get("analytics", {}).get("ai_confidence")
        if v is not None:
            return {"mode": "offline", "answer": str(v), "confidence": 1.0, "source": "local_fallback", "chunks_used": 0}
    if any(k in q for k in ["name", "names", "holder name", "person", "party", "signer"]):
        names = _collect_name_candidates(doc_text or "", a)
        if names:
            return {
                "mode": "offline",
                "answer": "Names found: " + ", ".join(names[:8]),
                "confidence": 0.9,
                "source": "local_fallback",
                "chunks_used": 0
            }
        return {
            "mode": "offline",
            "answer": "No clear person names detected in the document.",
            "confidence": 0.7,
            "source": "local_fallback",
            "chunks_used": 0
        }
    try:
        size = int(os.environ.get("RAG_CHUNK_SIZE", str(RAG_CHUNK_SIZE)))
        ov = int(os.environ.get("RAG_CHUNK_OVERLAP", str(RAG_CHUNK_OVERLAP)))
        k = int(os.environ.get("RAG_TOP_K", str(RAG_TOP_K)))
        chunks = chunk_document(doc_text or "", chunk_size=size, overlap=ov)
        vecs = embed_chunks(chunks) if chunks else []
        tops = retrieve_top_chunks(question, chunks, vecs, top_k=k) if chunks and vecs else []
        ans, conf = extract_answer(question, tops)
        return {"mode": "offline", "answer": ans, "confidence": float(conf), "source": "offline_rag", "chunks_used": len(tops or [])}
    except Exception:
        try:
            from backend.app.services.analysis_service import answer_with_fallback
            fb = answer_with_fallback(question, (doc_text or "")[:15000])
            return {"mode": "offline", "answer": fb, "confidence": 0.5, "source": "local_fallback", "chunks_used": 0}
        except Exception:
            return {"mode": "offline", "answer": "", "confidence": 0.0, "source": "local_fallback", "chunks_used": 0}
