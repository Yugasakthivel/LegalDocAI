import os
import torch
import re
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class SummarizationService:
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SummarizationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        # Legal-specific summarization model
        self.model_name = "nsandhu/legal-t5-base" 
        self.max_input_length = 512
        self.summary_max_length = 150
        self.summary_min_length = 40

    def _load_resources(self):
        if self._pipeline is None:
            try:
                print(f"[SummarizationService] Loading model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._pipeline = pipeline(
                    "summarization",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=self.device
                )
            except Exception as e:
                print(f"[SummarizationService] Error loading model: {e}")
                return None
        return self._pipeline

    def text_rank_summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        High-speed extractive summarization fallback using simplified TextRank logic.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= num_sentences:
            return text

        # Simplified sentence scoring based on keyword overlap
        scores = np.zeros(len(sentences))
        words = [set(s.lower().split()) for s in sentences]
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j: continue
                intersection = words[i].intersection(words[j])
                if intersection:
                    scores[i] += len(intersection) / (np.log(len(words[i])) + np.log(len(words[j])) + 1)

        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices.sort() # Keep original order
        
        return " ".join([sentences[i] for i in top_indices])

    def summarize_chunked(self, text: str) -> Dict[str, Any]:
        """
        Map-Reduce summarization for long documents.
        """
        # 1. Cleaning
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # 2. Check length - if short, use TextRank or simple return
        if len(clean_text) < 200:
            return {"summary": clean_text, "method": "none", "fallback": False}

        # 3. Strategy Decision
        use_fallback = False
        method = "LegalT5"
        
        try:
            nlp_pipe = self._load_resources()
            if not nlp_pipe:
                raise RuntimeError("Model not available")

            # Chunking for Map-Reduce (approx 400 words per chunk)
            words = clean_text.split()
            chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]
            
            chunk_summaries = []
            # Map step
            for chunk in chunks[:5]: # Limit to first 5 chunks for performance
                res = nlp_pipe(
                    chunk, 
                    max_length=self.summary_max_length, 
                    min_length=self.summary_min_length, 
                    do_sample=False,
                    no_repeat_ngram_size=3
                )
                chunk_summaries.append(res[0]['summary_text'])
            
            # Reduce step
            if len(chunk_summaries) > 1:
                combined = " ".join(chunk_summaries)
                final_res = nlp_pipe(
                    combined[:1000], 
                    max_length=self.summary_max_length + 50,
                    min_length=self.summary_min_length + 20,
                    do_sample=False
                )
                final_summary = final_res[0]['summary_text']
            else:
                final_summary = chunk_summaries[0]

        except Exception as e:
            print(f"[SummarizationService] BERT Summarization failed, using TextRank: {e}")
            final_summary = self.text_rank_summarize(clean_text)
            method = "TextRank"
            use_fallback = True

        # 4. Post-processing: Numeric consistency check
        # Ensure numbers in summary also exist in source
        original_numbers = set(re.findall(r'\d+', clean_text))
        summary_numbers = re.findall(r'\d+', final_summary)
        hallucinated_numbers = [n for n in summary_numbers if n not in original_numbers]
        
        if hallucinated_numbers:
            # If numbers are hallucinated, we prefer TextRank for safety
            final_summary = self.text_rank_summarize(clean_text)
            method = "TextRank (Hallucination Guard)"
            use_fallback = True

        return {
            "summary": final_summary,
            "method": method,
            "fallback": use_fallback,
            "metadata": {
                "model": self.model_name if not use_fallback else "extractive-textrank",
                "compression_ratio": round(len(final_summary) / max(1, len(clean_text)), 2)
            }
        }

summarization_service = SummarizationService()
