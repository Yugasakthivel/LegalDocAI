from backend.app.services.analysis_service import classify_legal_two_step

def test_resume_non_legal():
    text = "Resume\nEducation\nExperience\nSkills\nProjects\nSummary"
    res = classify_legal_two_step(text)
    assert res["document_type"] == "Resume"
    assert res["is_legal"] is False
    assert res["is_indian"] is False

def test_invoice_non_legal():
    text = "Tax Invoice\nInvoice No: INV-123\nGSTIN: 22AAAAA0000A1Z5\nBill To: ACME\nSubtotal\nTotal\nAmount Due"
    res = classify_legal_two_step(text)
    assert res["document_type"] == "Invoice"
    assert res["is_legal"] is False
    assert res["is_indian"] is False

def test_rental_agreement_indian_legal():
    text = "Rental Agreement between parties. Landlord and Tenant agree monthly rent and security deposit. Stamp Duty paid. Jurisdiction India. Rs. 5000."
    res = classify_legal_two_step(text)
    assert res["is_legal"] is True
    assert res["is_indian"] is True
    assert res["document_type"] in ["Rental Agreement", "Legal Agreement", "Contract"]

def test_affidavit_indian_legal():
    text = "Affidavit of the deponent before Notary. Witness signatures present. Jurisdiction India."
    res = classify_legal_two_step(text)
    assert res["is_legal"] is True
    assert res["is_indian"] is True
    assert res["document_type"] in ["Affidavit", "Legal Document"]

def test_sale_deed_indian_legal():
    text = "SALE DEED executed between Vendor and Vendee regarding schedule property. Stamp Duty and registration at Sub-Registrar office in India."
    res = classify_legal_two_step(text)
    assert res["is_legal"] is True
    assert res["is_indian"] is True
    assert res["document_type"] == "Sale Deed"

def test_court_order_indian_legal():
    text = "IN THE HIGH COURT OF XYZ. Court Order is hereby passed in favor of the petitioner against the respondent."
    res = classify_legal_two_step(text)
    assert res["is_legal"] is True
    assert res["is_indian"] is True
    assert res["document_type"] in ["Court Order", "Court Judgment"]

def test_partnership_deed_indian_legal():
    text = "Partnership Deed executed between partners forming a firm with profit sharing ratio specified."
    res = classify_legal_two_step(text)
    assert res["is_legal"] is True
    assert res["is_indian"] is True
    assert res["document_type"] == "Partnership Deed"
