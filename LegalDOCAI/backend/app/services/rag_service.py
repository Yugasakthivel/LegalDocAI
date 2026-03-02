import os
import re
import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from backend.app.core.deps import get_openai_client, is_openai_ready
from backend.app.core.config import OPENAI_MODEL

class LegalRAGService:
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LegalRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model_name = "paraphrase-MiniLM-L3-v2"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.top_k = 4
        self.score_threshold = 0.4
        self._load_embedding_model()

    def _load_embedding_model(self):
        if self._embedding_model is None:
            try:
                print(f"[LegalRAGService] Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            except Exception as e:
                print(f"[LegalRAGService] Error loading embedding model: {e}")

    def chunk_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        chunk_documents(): Splits documents into overlapping chunks for retrieval.
        """
        chunks = []
        chunk_id = 0
        for doc_idx, doc_text in enumerate(documents):
            # Clean text
            text = re.sub(r'\s+', ' ', doc_text).strip()
            # Simple fixed-size chunking with overlap
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_text = " ".join(words[i : i + self.chunk_size])
                if len(chunk_text.strip()) > 50: # Ignore tiny fragments
                    chunks.append({
                        "id": chunk_id,
                        "doc_index": doc_idx,
                        "text": chunk_text
                    })
                    chunk_id += 1
        return chunks

    def build_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        build_embeddings(): Generates normalized S-BERT embeddings for text list.
        """
        if not self._embedding_model:
            return np.array([])
        embeddings = self._embedding_model.encode(texts, convert_to_tensor=False)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        build_faiss_index(): Creates a FAISS index for efficient similarity search.
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine Similarity
        index.add(embeddings.astype('float32'))
        return index

    def retrieve_top_k(self, query: str, index: faiss.Index, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        retrieve_top_k(): Retrieves relevant document chunks for a query.
        """
        query_embedding = self.build_embeddings([query])
        scores, indices = index.search(query_embedding.astype('float32'), self.top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= self.score_threshold:
                chunk = chunks[idx].copy()
                chunk["similarity"] = float(score)
                results.append(chunk)
        return results

    def build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        build_rag_prompt(): Constructs a grounded prompt for the LLM.
        """
        context_text = "\n\n".join([f"Source {i+1}:\n{c['text']}" for i, c in enumerate(context_chunks)])
        
        prompt = (
            "You are a professional legal assistant. Answer the user's question STRICTLY based on the provided document context. "
            "If the answer is not contained in the context, state that 'The information is not found in the uploaded documents.' "
            "Do not use external knowledge. Be concise and accurate.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context_text}\n\n"
            f"--- QUESTION ---\n{query}\n\n"
            "--- ANSWER ---"
        )
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        generate_answer(): Calls the LLM (OpenAI or local) to generate a response.
        """
        if is_openai_ready():
            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, # Deterministic
                    max_tokens=500
                )
                return response.choices[0].message.content or "Error generating answer."
            except Exception as e:
                return f"LLM Error: {str(e)}"
        else:
            # Offline Fallback (Basic extractive logic if no LLM)
            return "Answer unavailable (LLM offline). Please check document sources below."

    def compute_confidence(self, retrieval_results: List[Dict[str, Any]], answer: str) -> float:
        """
        compute_confidence(): Heuristic confidence score based on retrieval similarity and answer quality.
        """
        if not retrieval_results:
            return 0.0
        
        # 1. Max retrieval similarity (0 to 1)
        max_sim = max([r["similarity"] for r in retrieval_results])
        
        # 2. Answer length/validity factor
        not_found_patterns = ["not found", "insufficient info", "unavailable", "does not contain"]
        if any(p in answer.lower() for p in not_found_patterns):
            return 0.0
            
        # 3. Context coverage (how many chunks were high-confidence)
        coverage = len([r for r in retrieval_results if r["similarity"] > 0.6]) / self.top_k
        
        confidence = (max_sim * 0.7) + (coverage * 0.3)
        return round(float(confidence), 2)

    def rag_pipeline(self, documents: List[str], user_query: str) -> Dict[str, Any]:
        """
        rag_pipeline(): main() equivalent for the full RAG process.
        """
        if not documents or not user_query.strip():
            return {
                "answer": "No documents provided or empty query.",
                "confidence": 0.0,
                "sources": [],
                "retrieval_stats": {"num_chunks": 0, "top_k": self.top_k}
            }

        try:
            # 1. Chunking
            chunks = self.chunk_documents(documents)
            if not chunks:
                return {"answer": "Documents are too short or empty.", "confidence": 0.0, "sources": []}

            # 2. Indexing
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.build_embeddings(chunk_texts)
            index = self.build_faiss_index(embeddings)

            # 3. Retrieval
            retrieved_chunks = self.retrieve_top_k(user_query, index, chunks)
            
            # 4. Generation
            if not retrieved_chunks:
                answer = "The information is not found in the uploaded documents."
                confidence = 0.0
            else:
                prompt = self.build_rag_prompt(user_query, retrieved_chunks)
                answer = self.generate_answer(prompt)
                confidence = self.compute_confidence(retrieved_chunks, answer)

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": retrieved_chunks,
                "retrieval_stats": {
                    "num_chunks": len(chunks),
                    "top_k": self.top_k,
                    "embedding_model": self.embedding_model_name,
                    "llm_model": OPENAI_MODEL if is_openai_ready() else "offline-extractive"
                }
            }
        except Exception as e:
            print(f"[LegalRAGService] Pipeline Error: {e}")
            return {
                "answer": f"System error during RAG analysis: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }

# Singleton instance
legal_rag_service = LegalRAGService()
