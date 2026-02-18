from backend.app.services.analysis_service import classify_document_type, extract_indian_entities

def test_regulatory_filing_cin_detection():
    text = "REGULATORY FILING submitted to ROC. CIN: L12345MH2010PLC123456. DIN: 01234567."
    doc_type = classify_document_type(text)
    assert doc_type == "Regulatory Filing"
    ents = extract_indian_entities(text)
    # Expect registration numbers to include CIN and DIN entries
    assert any("CIN" in r or ("L" in r and "PLC" in r) for r in ents["registration_numbers"]) or "L12345MH2010PLC123456" in " ".join(ents["registration_numbers"])
    assert any("DIN" in r for r in ents["registration_numbers"])
