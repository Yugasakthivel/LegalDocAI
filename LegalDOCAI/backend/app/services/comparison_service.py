import re
import difflib
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
try:
    import Levenshtein
except ImportError:
    Levenshtein = None

class ComparisonService:
    _instance = None
    _sbert_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComparisonService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_sbert()
        # Thresholds for classification
        self.similarity_threshold = 0.95  # Above this = unchanged
        self.modified_threshold = 0.40    # Below this = added/removed (new clause)

    def _load_sbert(self):
        if self._sbert_model is None:
            try:
                # Use a fast, high-quality model for clause embeddings
                self._sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
            except Exception as e:
                print(f"[ComparisonService] Error loading S-BERT: {e}")

    def split_into_clauses(self, text: str) -> List[str]:
        """
        split_into_clauses(): Splits document into logical clauses based on newlines and legal numbering.
        """
        if not text:
            return []
        # Split by double newlines or legal numbering patterns
        segments = re.split(r'\n\s*\n|(?m)^(?:\d+\.[\d\.]*|\([a-z0-9]+\)|[ivx]+\.)\s+', text)
        return [s.strip() for s in segments if len(s.strip()) > 5]

    def compute_levenshtein(self, s1: str, s2: str) -> float:
        """
        compute_levenshtein(): Calculates normalized Levenshtein similarity score (0.0 to 1.0).
        """
        if not s1 or not s2:
            return 0.0
        if Levenshtein:
            distance = Levenshtein.distance(s1, s2)
            max_len = max(len(s1), len(s2))
            return 1.0 - (distance / max_len)
        else:
            # Fallback to difflib
            return difflib.SequenceMatcher(None, s1, s2).ratio()

    def compute_semantic_similarity(self, s1: str, s2: str) -> float:
        """
        compute_semantic_similarity(): Calculates cosine similarity using S-BERT.
        """
        if not self._sbert_model or not s1 or not s2:
            return 0.0
        emb1 = self._sbert_model.encode(s1, convert_to_tensor=True)
        emb2 = self._sbert_model.encode(s2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0].item())

    def generate_diff_html(self, original: str, revised: str) -> str:
        """
        generate_diff_html(): Generates HTML with green (insertion) and red (deletion) highlighting.
        """
        if not original:
            return f'<span style="background-color: #d4edda; color: #155724;">{revised}</span>'
        if not revised:
            return f'<span style="background-color: #f8d7da; color: #721c24;">{original}</span>'

        output = []
        matcher = difflib.SequenceMatcher(None, original, revised)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                output.append(original[i1:i2])
            elif tag == 'insert':
                output.append(f'<span style="background-color: #d4edda; color: #155724;">{revised[j1:j2]}</span>')
            elif tag == 'delete':
                output.append(f'<span style="background-color: #f8d7da; color: #721c24;">{original[i1:i2]}</span>')
            elif tag == 'replace':
                output.append(f'<span style="background-color: #f8d7da; color: #721c24;">{original[i1:i2]}</span>')
                output.append(f'<span style="background-color: #d4edda; color: #155724;">{revised[j1:j2]}</span>')
        
        return "".join(output)

    def classify_change(self, lev_score: float, sem_score: float) -> str:
        """
        classify_change(): Determines change type based on similarity thresholds.
        """
        avg_score = (lev_score + sem_score) / 2
        if avg_score >= self.similarity_threshold:
            return "unchanged"
        elif avg_score >= self.modified_threshold:
            return "modified"
        else:
            return "added" # This will be refined during alignment

    def align_clauses(self, original_clauses: List[str], revised_clauses: List[str]) -> List[Dict[str, Any]]:
        """
        align_clauses(): Aligns clauses between two versions using hybrid similarity.
        """
        results = []
        used_revised = set()

        for i, orig in enumerate(original_clauses):
            best_match_idx = -1
            max_sim = -1.0
            
            # Find best match in revised version
            for j, rev in enumerate(revised_clauses):
                if j in used_revised:
                    continue
                
                # Fast check with Levenshtein
                lev = self.compute_levenshtein(orig, rev)
                if lev > 0.8: # If very similar, skip semantic for speed
                    sim = lev
                else:
                    sem = self.compute_semantic_similarity(orig, rev)
                    sim = (lev + sem) / 2

                if sim > max_sim:
                    max_sim = sim
                    best_match_idx = j

            # Decision logic
            if best_match_idx != -1 and max_sim >= self.modified_threshold:
                used_revised.add(best_match_idx)
                rev_text = revised_clauses[best_match_idx]
                lev_score = self.compute_levenshtein(orig, rev_text)
                sem_score = self.compute_semantic_similarity(orig, rev_text)
                
                change_type = "unchanged" if max_sim >= self.similarity_threshold else "modified"
                
                results.append({
                    "clause_id": i,
                    "original": orig,
                    "revised": rev_text,
                    "change_type": change_type,
                    "levenshtein_score": round(lev_score, 4),
                    "semantic_similarity": round(sem_score, 4),
                    "diff_html": self.generate_diff_html(orig, rev_text) if change_type == "modified" else orig
                })
            else:
                # Removed clause
                results.append({
                    "clause_id": i,
                    "original": orig,
                    "revised": "",
                    "change_type": "removed",
                    "levenshtein_score": 0.0,
                    "semantic_similarity": 0.0,
                    "diff_html": self.generate_diff_html(orig, "")
                })

        # Add remaining revised clauses as "added"
        for j, rev in enumerate(revised_clauses):
            if j not in used_revised:
                results.append({
                    "clause_id": len(results),
                    "original": "",
                    "revised": rev,
                    "change_type": "added",
                    "levenshtein_score": 0.0,
                    "semantic_similarity": 0.0,
                    "diff_html": self.generate_diff_html("", rev)
                })

        return results

    def compute_overall_score(self, clause_changes: List[Dict[str, Any]]) -> float:
        """
        compute_overall_score(): Calculates percentage of document that has changed.
        """
        if not clause_changes:
            return 0.0
        
        changed_count = sum(1 for c in clause_changes if c["change_type"] != "unchanged")
        return round((changed_count / len(clause_changes)) * 100, 2)

    def compare_documents(self, original_text: str, revised_text: str) -> Dict[str, Any]:
        """
        main(): Full document comparison pipeline.
        """
        if not original_text and not revised_text:
            return {"overall_change_score": 0.0, "summary": {"added": 0, "removed": 0, "modified": 0}, "clause_changes": []}

        # 1. Preprocessing & Segmentation
        orig_clauses = self.split_into_clauses(original_text)
        rev_clauses = self.split_into_clauses(revised_text)

        # 2. Alignment & Analysis
        clause_changes = self.align_clauses(orig_clauses, rev_clauses)

        # 3. Summarization
        added = sum(1 for c in clause_changes if c["change_type"] == "added")
        removed = sum(1 for c in clause_changes if c["change_type"] == "removed")
        modified = sum(1 for c in clause_changes if c["change_type"] == "modified")
        
        overall_score = self.compute_overall_score(clause_changes)

        return {
            "overall_change_score": overall_score,
            "summary": {
                "added": added,
                "removed": removed,
                "modified": modified
            },
            "clause_changes": clause_changes
        }

# Singleton instance
comparison_service = ComparisonService()
