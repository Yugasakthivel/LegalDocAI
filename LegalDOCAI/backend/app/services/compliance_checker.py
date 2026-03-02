import json
import os
import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from backend.app.core.deps import get_async_openai_client, is_openai_ready

class ComplianceService:
    _instance = None
    _rules = []
    _sbert_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComplianceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rules_path = os.path.join("d:\\Legal-mohan\\Legaldoc-new\\LegalDOCAI\\data", "regulatory_rules.json")
        self._load_rules()
        self._load_sbert()

    def _load_rules(self):
        """
        load_rules(): Loads the regulatory rules from the JSON knowledge base.
        """
        try:
            if os.path.exists(self.rules_path):
                with open(self.rules_path, 'r') as f:
                    data = json.load(f)
                    self._rules = data.get("rules", [])
            else:
                print(f"[ComplianceService] Warning: Rules file not found at {self.rules_path}")
        except Exception as e:
            print(f"[ComplianceService] Error loading rules: {e}")

    def _load_sbert(self):
        if self._sbert_model is None:
            try:
                self._sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
            except Exception as e:
                print(f"[ComplianceService] Error loading S-BERT: {e}")

    def check_rule_based(self, text: str, rule: Dict[str, Any]) -> bool:
        """
        check_rule_based(): Performs keyword-based clause presence checking.
        """
        text_lower = text.lower()
        for keyword in rule.get("keywords", []):
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                return True
        return False

    def semantic_match(self, text: str, rule: Dict[str, Any]) -> float:
        """
        semantic_match(): Performs semantic matching using S-BERT for complex validation.
        """
        if not self._sbert_model or not text:
            return 0.0
        
        # We split the text into chunks for better semantic matching
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        query_embedding = self._sbert_model.encode(rule.get("semantic_description", ""), convert_to_tensor=True)
        chunk_embeddings = self._sbert_model.encode(chunks, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        return float(torch.max(cosine_scores).item())

    def compute_compliance_score(self, results: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        compute_compliance_score(): Computes the final compliance score and risk level.
        """
        total_weight = sum(r["weight"] for r in self._rules)
        if total_weight == 0:
            return 100.0, "Low"
            
        earned_score = 0
        for res in results:
            if res["compliant"]:
                earned_score += res["weight"]
            elif res["partial_match"]:
                # Partial credit for semantic matches that didn't hit the threshold
                earned_score += res["weight"] * 0.5

        final_score = (earned_score / total_weight) * 100
        
        risk_level = "High"
        if final_score >= 80:
            risk_level = "Low"
        elif final_score >= 50:
            risk_level = "Medium"
            
        return round(final_score, 2), risk_level

    async def check_compliance(self, text: str) -> Dict[str, Any]:
        """
        main() equivalent: Full compliance analysis pipeline.
        """
        if not text:
            return {
                "compliance_score": 0.0,
                "violations": ["Empty document"],
                "risk_level": "High",
                "recommendations": ["Provide a valid document for analysis."]
            }

        results = []
        violations = []
        recommendations = []

        for rule in self._rules:
            # 1. Rule-based check
            is_rule_compliant = self.check_rule_based(text, rule)
            
            # 2. Semantic match check
            semantic_score = self.semantic_match(text, rule)
            is_semantic_compliant = semantic_score > 0.55
            
            compliant = is_rule_compliant or is_semantic_compliant
            partial_match = not compliant and semantic_score > 0.35
            
            results.append({
                "clause_name": rule["clause_name"],
                "compliant": compliant,
                "partial_match": partial_match,
                "weight": rule["weight"]
            })

            if not compliant:
                violations.append(f"Missing or non-compliant {rule['clause_name']}")
                recommendations.append(rule["recommendation"])

        # 3. Final Scoring
        score, risk = self.compute_compliance_score(results)

        return {
            "compliance_score": score,
            "violations": violations,
            "risk_level": risk,
            "recommendations": list(set(recommendations)),
            "metadata": {
                "engine": "RegulatoryCompliance-Hybrid-v1.0",
                "framework": "Indian Legal Standards"
            }
        }

compliance_service = ComplianceService()
