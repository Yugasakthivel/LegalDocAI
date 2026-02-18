from backend.app.services.analysis_service import extract_indian_entities

def test_extract_sale_deed_entities():
    text = "SALE DEED between Vendor and Vendee for schedule property. Registration No: 123/2020 at Sub-Registrar Office. Stamp Duty Rs. 10,000. Survey No: 45/2."
    e = extract_indian_entities(text)
    assert "Vendor" in " ".join(e["parties"]) or "vendor" in " ".join(e["role_parties"])
    assert any("Registration" in r or "reg" in r.lower() for r in e["registration_numbers"])
    assert any("Survey" in s or "S.No" in s for s in e["survey_numbers"])
    assert any("Rs" in m or "₹" in m for m in e["money"])
    assert any("sub-registrar" in c.lower() for c in e["courts"])
    assert e["property_details"] != ""

def test_extract_court_order_entities():
    text = "IN THE HIGH COURT OF MADRAS. Case No: WP-1234/2024. Court Order is passed. Petitioner and Respondent present."
    e = extract_indian_entities(text)
    assert any("high court" in c.lower() for c in e["courts"])
    assert any("case" in cn.lower() or "wp" in cn.lower() for cn in e["case_numbers"])
    assert "petitioner" in " ".join(e["role_parties"]) or "respondent" in " ".join(e["role_parties"])

def test_extract_rental_agreement_entities():
    text = "Rental Agreement between Landlord and Tenant. Monthly rent ₹ 15,000. Lease term 11 months."
    e = extract_indian_entities(text)
    assert "landlord" in " ".join(e["role_parties"]) and "tenant" in " ".join(e["role_parties"])
    assert any("₹" in m or "rs" in m.lower() for m in e["money"])
