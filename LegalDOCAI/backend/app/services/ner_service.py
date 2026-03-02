import os
import torch
import spacy
import re
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class NERService:
    _instance = None
    _models = {}
    _pipelines = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NERService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = "nlpaueb/legal-bert-base-uncased" # Or a fine-tuned legal NER model
        # Note: nlpaueb/legal-bert-base-uncased is a base model, 
        # for NER we typically use models like "saibo/legal-roberta-base-ner-legal"
        # or custom fine-tuned models.
        self.ner_model_name = "saibo/legal-roberta-base-ner-legal" 

    def _get_pipeline(self, model_name: str):
        if model_name not in self._pipelines:
            try:
                print(f"[NERService] Loading model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self._pipelines[model_name] = pipeline(
                    "ner", 
                    model=model, 
                    tokenizer=tokenizer, 
                    device=self.device,
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"[NERService] Error loading model {model_name}: {e}")
                return None
        return self._pipelines[model_name]

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities using a legal-specific BERT model.
        """
        if not text or not text.strip():
            return {"entities": [], "model": "none"}

        nlp_pipeline = self._get_pipeline(self.ner_model_name)
        if not nlp_pipeline:
            return {"entities": [], "error": "Model failed to load"}

        # Process text in chunks if too long for BERT (max 512 tokens)
        # For simplicity, we'll just truncate or use simple chunking
        max_len = 512
        results = nlp_pipeline(text[:2000]) # Example limit

        # Post-processing and normalization
        processed_entities = []
        for ent in results:
            processed_entities.append({
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": round(float(ent["score"]), 4),
                "start": ent["start"],
                "end": ent["end"]
            })

        return {
            "entities": processed_entities,
            "model": self.ner_model_name,
            "version": "1.0.0"
        }

    def normalize_money(self, text: str) -> Optional[float]:
        """
        Normalizes Indian currency formats (e.g., Rs. 50,000, INR 5 Lakhs).
        """
        # Remove currency symbols and commas
        clean = re.sub(r'[₹Rs\.INR,]', '', text, flags=re.IGNORECASE).strip()
        
        # Handle "Lakh" and "Crore"
        multiplier = 1.0
        if "lakh" in clean.lower():
            multiplier = 100000.0
            clean = re.sub(r'lakh[s]?', '', clean, flags=re.IGNORECASE).strip()
        elif "crore" in clean.lower():
            multiplier = 10000000.0
            clean = re.sub(r'crore[s]?', '', clean, flags=re.IGNORECASE).strip()
            
        try:
            return float(clean) * multiplier
        except ValueError:
            return None

    def normalize_date(self, text: str) -> Optional[str]:
        """
        Normalizes dates into YYYY-MM-DD.
        """
        import dateparser
        dt = dateparser.parse(text, settings={'PREFER_DAY_OF_MONTH': 'first'})
        if dt:
            return dt.strftime("%Y-%m-%d")
        return None

ner_service = NERService()
