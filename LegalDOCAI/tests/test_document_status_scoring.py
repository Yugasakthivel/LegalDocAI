from backend.app.services import analysis_service


def test_compute_genuineness_score_bounds_and_type():
    analytics = {
        "ai_confidence": 82,
        "mandatory_fields": {
            "present": ["Signature", "Parties", "Governing Law", "Payment Terms"],
            "missing": ["Warranty"],
        },
        "clause_risk": {"overall_risk_score": 30},
        "signer_validation": {"overall_signer_score": 78},
    }
    score = analysis_service.compute_genuineness_score(analytics, {"ai_confidence": 82})
    assert isinstance(score, int)
    assert 0 <= score <= 100


def test_derive_legal_status_not_legal_when_rejected():
    analytics = {
        "document_type": "Aadhaar Card",
        "legality_score": 44,
        "genuineness_score": 30,
        "verified_marker": "Rejected",
    }
    status = analysis_service.derive_legal_status(analytics, {"marker": "Rejected", "risk_level": "Critical"})
    assert status["status"] == "NOT_LEGAL"


def test_derive_legal_status_legal_when_scores_good():
    analytics = {
        "document_type": "Passport",
        "legality_score": 78,
        "genuineness_score": 84,
        "verified_marker": "Verified",
    }
    status = analysis_service.derive_legal_status(analytics, {"marker": "Verified", "risk_level": "Low"})
    assert status["status"] == "LEGAL"


def test_build_required_output_schema_contract_keys_present():
    results = {
        "summary": "Basic summary",
        "entities": {"names": ["Arun"], "organizations": ["TN Registry"]},
        "contact_info": {"emails": ["a@example.com"], "phones": [{"number": "+91 9876543210"}]},
        "dates": ["2024-10-12"],
        "clauses_found": {"termination": 1, "liability": 0},
        "metadata": {
            "survey_numbers": ["45/2A"],
            "case_numbers": [],
            "authorities": ["Sub-Registrar"],
            "keyword_frequency": {"patta": 3, "survey": 2},
            "land_classification": {"primary_land_type": "Nanjai", "land_types_detected": ["Nanjai"]},
        },
    }
    analytics = {
        "document_type": "Patta",
        "clause_risk": {
            "overall_risk": "Medium",
            "overall_risk_score": 48,
            "clauses": [{"clause": "Termination", "risk": "High", "reason": "Can terminate anytime"}],
        },
        "combined_risk_index": 52,
        "legal_status": "NEEDS_REVIEW",
        "mandatory_fields": {"missing": ["Signature"], "present": ["Parties"]},
        "registration_analysis": {"missing_fields": ["Registration Number"]},
        "jurisdiction_curated": {"checks": [{"rule": "Governing Law", "status": "fail", "severity": "Medium"}]},
        "genuineness_score": 67,
        "status_summary": "Status summary",
        "land_classification": {"primary_land_type": "Nanjai", "land_types_detected": ["Nanjai"]},
    }

    out = analysis_service.build_required_output_schema(results, analytics, "sample legal text")
    required_keys = {
        "document_type",
        "summary_simple",
        "entities",
        "clauses",
        "key_terms",
        "risk_analysis",
        "compliance_flags",
        "confidence_score",
    }
    assert required_keys.issubset(set(out.keys()))
    assert out["document_type"] == "Patta"
    assert isinstance(out["compliance_flags"], list)
