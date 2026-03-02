import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util
import torch
import re
from typing import List, Dict, Any, Optional, Tuple

class ClausePresenceService:
    _instance = None
    _nlp = None
    _matcher = None
    _sbert_model = None

    # Define standard clauses and their semantic descriptions for S-BERT
    CLAUSE_DEFINITIONS = {
        "Governing Law": {
            "patterns": [
                [{"LEMMA": "govern"}, {"LEMMA": "law"}],
                [{"LEMMA": "jurisdiction"}],
                [{"LOWER": "applicable"}, {"LOWER": "law"}]
            ],
            "semantic_query": "The agreement is governed by the laws of a specific state or country."
        },
        "Signature": {
            "patterns": [
                [{"LEMMA": "signature"}],
                [{"LOWER": "signed"}, {"LOWER": "by"}],
                [{"LOWER": "authorized"}, {"LOWER": "signatory"}]
            ],
            "semantic_query": "Signatures of the authorized persons or representatives."
        },
        "Effective Date": {
            "patterns": [
                [{"LOWER": "effective"}, {"LOWER": "date"}],
                [{"LEMMA": "commence"}],
                [{"LOWER": "date"}, {"LOWER": "of"}, {"LOWER": "agreement"}]
            ],
            "semantic_query": "The date when this agreement becomes legally binding."
        },
        "Payment Terms": {
            "patterns": [
                [{"LEMMA": "payment"}],
                [{"LEMMA": "consideration"}],
                [{"LEMMA": "fee"}],
                [{"LEMMA": "charge"}]
            ],
            "semantic_query": "Terms related to payment, consideration, fees, and currency."
        },
        "Termination": {
            "patterns": [
                [{"LEMMA": "termination"}],
                [{"LEMMA": "terminate"}],
                [{"LOWER": "end"}, {"LOWER": "of"}, {"LOWER": "term"}]
            ],
            "semantic_query": "Conditions under which the agreement can be terminated or cancelled."
        },
        "Confidentiality": {
            "patterns": [
                [{"LEMMA": "confidentiality"}],
                [{"LOWER": "non-disclosure"}],
                [{"LOWER": "secret"}, {"LOWER": "information"}]
            ],
            "semantic_query": "Obligations to keep sensitive information private and confidential."
        },
        "Dispute Resolution": {
            "patterns": [
                [{"LEMMA": "arbitration"}],
                [{"LEMMA": "dispute"}, {"LEMMA": "resolution"}],
                [{"LEMMA": "conciliation"}],
                [{"LEMMA": "mediation"}]
            ],
            "semantic_query": "Mechanism for resolving disputes such as arbitration, mediation or court."
        }
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClausePresenceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_resources()

    def _load_resources(self):
        if self._nlp is None:
            try:
                # Use a standard spaCy model
                self._nlp = spacy.load("en_core_web_sm")
                self._matcher = Matcher(self._nlp.vocab)
                
                # Add patterns to matcher
                for clause_name, definition in self.CLAUSE_DEFINITIONS.items():
                    self._matcher.add(clause_name, definition["patterns"])
            except Exception as e:
                print(f"[ClausePresenceService] Error loading spaCy: {e}")

        if self._sbert_model is None:
            try:
                # Use a lightweight S-BERT model
                self._sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
            except Exception as e:
                print(f"[ClausePresenceService] Error loading S-BERT: {e}")

    def segment_clauses(self, text: str) -> List[str]:
        """
        Segments text into potential clause blocks.
        """
        # Simple segmentation by double newlines or legal numbering
        segments = re.split(r'\n\n|(?m)^(?:\d+\.[\d\.]*|\([a-z0-9]+\)|[ivx]+\.)\s+', text)
        return [s.strip() for s in segments if len(s.strip()) > 20]

    def check_presence(self, text: str) -> Dict[str, Any]:
        """
        Main pipeline for Module 9: Clause Presence Checker.
        """
        if not text:
            return {"present": [], "missing": [], "details": {}}

        # 1. Clause Segmentation
        segments = self.segment_clauses(text)
        if not segments:
            segments = [text] # Fallback

        doc = self._nlp(text)
        
        # 2. Rule-based detection
        matches = self._matcher(doc)
        rule_results = {}
        for match_id, start, end in matches:
            clause_name = self._nlp.vocab.strings[match_id]
            if clause_name not in rule_results:
                rule_results[clause_name] = []
            rule_results[clause_name].append(doc[start:end].text)

        # 3. Semantic detection (S-BERT)
        semantic_results = {}
        if segments and self._sbert_model:
            # Encode segments and queries
            segment_embeddings = self._sbert_model.encode(segments, convert_to_tensor=True)
            
            for clause_name, definition in self.CLAUSE_DEFINITIONS.items():
                query_embedding = self._sbert_model.encode(definition["semantic_query"], convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_embedding, segment_embeddings)[0]
                
                # Find best matching segment
                best_score_idx = torch.argmax(cosine_scores).item()
                best_score = cosine_scores[best_score_idx].item()
                
                if best_score > 0.45: # Threshold calibration
                    semantic_results[clause_name] = {
                        "score": round(best_score, 4),
                        "text": segments[best_score_idx]
                    }

        # 4. Hybrid Fusion Logic
        final_presence = []
        final_missing = []
        details = {}

        for clause_name in self.CLAUSE_DEFINITIONS.keys():
            has_rule = clause_name in rule_results
            has_semantic = clause_name in semantic_results
            
            # Confidence scoring
            confidence = 0.0
            matched_text = ""
            method = ""

            if has_rule and has_semantic:
                confidence = max(0.9, semantic_results[clause_name]["score"])
                matched_text = semantic_results[clause_name]["text"]
                method = "Hybrid (Rule + Semantic)"
            elif has_rule:
                confidence = 0.75
                matched_text = rule_results[clause_name][0]
                method = "Rule-based"
            elif has_semantic:
                confidence = semantic_results[clause_name]["score"]
                matched_text = semantic_results[clause_name]["text"]
                method = "Semantic"

            # False Positive Controls: Negation handling
            if matched_text and re.search(r'\b(not|no|never|none|neither|nor)\b', matched_text.lower()[:50]):
                # If negation is found at the start of the matched text, reduce confidence
                confidence *= 0.5

            if confidence > 0.6:
                final_presence.append(clause_name)
                details[clause_name] = {
                    "present": True,
                    "confidence": round(confidence, 4),
                    "matched_text_snippet": matched_text[:200] + "...",
                    "method": method
                }
            else:
                final_missing.append(clause_name)
                details[clause_name] = {
                    "present": False,
                    "confidence": round(confidence, 4),
                    "method": method or "None"
                }

        return {
            "present": final_presence,
            "missing": final_missing,
            "details": details,
            "metadata": {
                "engine": "ClausePresence-Hybrid-v1.0",
                "model": "S-BERT-MiniLM-L3"
            }
        }

clause_presence_service = ClausePresenceService()
