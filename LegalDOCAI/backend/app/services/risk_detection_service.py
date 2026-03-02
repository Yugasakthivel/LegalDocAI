import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class RiskDetectionService:
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None

    # Risk Labels Schema
    RISK_LABELS = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Critical"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RiskDetectionService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        # Using a model suitable for legal text classification
        # In production, this would be a fine-tuned version for clause risk
        self.model_name = "nlpaueb/legal-bert-base-uncased" 

    def _load_resources(self):
        if self._model is None:
            print(f"[RiskDetectionService] Loading model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # We use a sequence classifier for multi-class risk labels
            # For this implementation, we'll use a generic legal-bert and simulate risk heads 
            # or use a pre-trained risk model if available.
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=4)
            self._pipeline = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device,
                return_all_scores=True
            )

    def segment_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Segments text into clauses using legal numbering and punctuation.
        """
        # Patterns for legal numbering: 1., 1.1, (a), (i), Article 1, etc.
        pattern = r'(?m)(?:^\s*(?:\d+\.[\d\.]*|\([a-z0-9]+\)|[ivx]+\.)\s+|^Article\s+\d+\s+|^Clause\s+\d+\s+)'
        
        # Split text into segments
        splits = list(re.finditer(pattern, text))
        
        clauses = []
        if not splits:
            # Fallback to double newline if no numbering found
            segments = [s.strip() for s in text.split('\n\n') if s.strip()]
            for i, seg in enumerate(segments):
                clauses.append({"id": f"c_{i}", "content": seg, "numbering": None})
        else:
            for i in range(len(splits)):
                start = splits[i].start()
                end = splits[i+1].start() if i+1 < len(splits) else len(text)
                content = text[start:end].strip()
                numbering = splits[i].group(0).strip()
                clauses.append({"id": f"c_{i}", "content": content, "numbering": numbering})
                
        return clauses

    def analyze_clause_risk(self, clause_text: str) -> Dict[str, Any]:
        """
        Classifies risk of a single clause and generates XAI data.
        """
        self._load_resources()
        
        # 1. Inference
        inputs = self._tokenizer(clause_text, return_tensors="pt", truncation=True, max_length=512).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions # List of (batch, num_heads, seq_len, seq_len)
            
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        risk_idx = np.argmax(probs)
        risk_level = self.RISK_LABELS[risk_idx]
        confidence = float(probs[risk_idx])

        # 2. XAI: Word-level importance (Simplified Attention-based)
        # Use last layer's average attention over all heads
        last_layer_attn = attentions[-1].cpu().numpy()[0] # (num_heads, seq_len, seq_len)
        avg_attn = np.mean(last_layer_attn, axis=0) # (seq_len, seq_len)
        
        # Get attention from [CLS] token to other tokens
        cls_attn = avg_attn[0, 1:-1] # Ignore [CLS] and [SEP]
        
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][1:-1])
        
        # Aggregate word-level scores (BERT tokens to words)
        word_scores = []
        for token, score in zip(tokens, cls_attn):
            if token.startswith("##"):
                if word_scores:
                    word_scores[-1]["score"] += float(score)
            else:
                word_scores.append({"word": token, "score": float(score)})
                
        # Normalize scores
        max_score = max([w["score"] for w in word_scores]) if word_scores else 1.0
        for w in word_scores:
            w["score"] = round(w["score"] / max_score, 4)

        return {
            "risk_level": risk_level,
            "risk_score": int(risk_idx * 33.3), # Scaled 0-100
            "confidence": round(confidence, 4),
            "explanation": {
                "top_words": sorted(word_scores, key=lambda x: x["score"], reverse=True)[:5],
                "highlights": word_scores
            }
        }

    def detect_all_risks(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline: segment -> analyze each -> aggregate.
        """
        clauses = self.segment_clauses(text)
        results = []
        overall_risk_score = 0
        
        for c in clauses:
            analysis = self.analyze_clause_risk(c["content"])
            c.update(analysis)
            results.append(c)
            overall_risk_score = max(overall_risk_score, analysis["risk_score"])
            
        overall_level = "Low"
        if overall_risk_score > 80: overall_level = "Critical"
        elif overall_risk_score > 60: overall_level = "High"
        elif overall_risk_score > 30: overall_level = "Medium"

        return {
            "overall_risk": overall_level,
            "overall_score": overall_risk_score,
            "clauses": results,
            "metadata": {
                "model": self.model_name,
                "engine": "LegalBERT-XAI"
            }
        }

risk_detection_service = RiskDetectionService()
