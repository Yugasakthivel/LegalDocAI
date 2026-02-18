import pytest
from backend.app.services import analysis_service

def test_document_type_fallback():
    # Test rule-based fallback when ML fails
    text = "This is a patta document with survey number 123/4"
    doc_type = analysis_service.classify_document_type(text)
    assert doc_type == "Patta"

def test_document_type_ml_fallback():
    # Test ML fallback when rule-based fails
    text = "This is a complex legal document with multiple clauses"
    doc_type = analysis_service.classify_document_type(text)
    assert isinstance(doc_type, str)
    assert len(doc_type.strip()) > 0

def test_legality_score_defaults():
    # Test default scores when components fail
    analytics = {
        "signer_validation": {},
        "jurisdiction_curated": {},
        "clause_risk": {},
        "mandatory_fields": {}
    }
    score = analysis_service.compute_legality_score(analytics)
    assert 0 <= score <= 100

def test_ai_detection_score_defaults():
    # Test default AI detection score
    text = "This is a short text"
    score = analysis_service.compute_ai_detection_score(text)
    assert score is None or (0 <= score <= 100)

    text = "This is a longer text with sufficient words to analyze"
    score = analysis_service.compute_ai_detection_score(text)
    assert 0 <= score <= 100

def test_genuineness_score_defaults():
    # Test default genuineness score
    analytics = {}
    score = analysis_service.compute_genuineness_score(analytics)
    assert 0 <= score <= 100

def test_output_schema_validation():
    # Test output schema validation
    results = {
        "summary": "Test summary",
        "entities": {"names": ["John"], "organizations": ["Company"]},
        "contact_info": {"emails": ["test@example.com"], "phones": [{"number": "+91 9876543210"}]},
        "dates": ["2024-01-01"],
        "clauses_found": {"termination": 1},
        "metadata": {
            "survey_numbers": ["123/4"],
            "case_numbers": [],
            "authorities": ["Court"],
            "keyword_frequency": {"test": 3},
            "land_classification": {"primary_land_type": "Unknown", "land_types_detected": []},
        },
    }
    analytics = {
        "document_type": "Test Document",
        "clause_risk": {
            "overall_risk": "Low",
            "overall_risk_score": 20,
            "clauses": [{"clause": "Termination", "risk": "Low", "reason": "Standard clause"}],
        },
        "combined_risk_index": 30,
        "legal_status": "LEGAL",
        "mandatory_fields": {"missing": [], "present": ["Signature"]},
        "genuineness_score": 85,
        "status_summary": "Test status",
        "land_classification": {"primary_land_type": "Unknown", "land_types_detected": []},
    }

    output = analysis_service.build_required_output_schema(results, analytics, "test text")
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
    assert required_keys.issubset(set(output.keys()))
    assert output["document_type"] == "Test Document"
    assert isinstance(output["compliance_flags"], list)
