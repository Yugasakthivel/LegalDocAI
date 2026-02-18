from backend.app.services.analysis_service import extract_indian_entities

def test_ustamp_uin_extraction():
    text = "e-Stamp Certificate. Unique Doc Ref: UIN-AB12CD3456. GRN: 9876543210. Stamp Duty Rs. 1,000."
    e = extract_indian_entities(text)
    assert any("Unique Doc Ref" in r or "UIN" in r for r in e["registration_numbers"])
    assert any("GRN" in r.upper() for r in e["registration_numbers"])
    assert any("Rs" in m or "₹" in m for m in e["money"])

def test_sro_code_extraction():
    text = "Registered at Sub-Registrar Office Chennai. SRO Code: CHN-123."
    e = extract_indian_entities(text)
    assert any("SRO" in r for r in e["registration_numbers"])
    assert any("sub-registrar" in c.lower() for c in e["courts"])
