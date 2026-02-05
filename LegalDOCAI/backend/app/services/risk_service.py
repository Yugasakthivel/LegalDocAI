import json
from typing import Dict, Any
from backend.app.services.analysis_service import clause_risk_detection as regex_risk_detection

def clause_risk_detection(full_text: str) -> Dict[str, Any]:
    """
    Delegate to the rule-based implementation in analysis_service.
    """
    return regex_risk_detection(full_text)

def compute_combined_risk_index(analytics: Dict[str, Any]) -> int:
    ls = float(analytics.get("legality_score") or 0)
    cr = analytics.get("clause_risk") or {}
    crs = float(cr.get("overall_risk_score") or 0)
    sv = analytics.get("signer_validation") or {}
    svs = float(sv.get("overall_signer_score") or 0)
    
    jc = analytics.get("jurisdiction_curated") or {}
    jcs = float(jc.get("overall_compliance_score") or 0)
    
    ai = float(analytics.get("ai_confidence") or 0)
    
    # Simple weighted average
    weights = [0.25, 0.2, 0.2, 0.2, 0.15]
    values = [ls, crs, svs, jcs, ai]
    
    total = sum(w * v for w, v in zip(weights, values))
    return int(round(min(100, max(0, total))))
