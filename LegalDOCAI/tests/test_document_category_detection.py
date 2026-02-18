from backend.app.services.analysis_service import classify_document_category

def test_sale_deed_category():
    text = "SALE DEED with vendor and vendee for schedule property and consideration."
    assert classify_document_category(text) == "Land/Property Document"

def test_patta_category():
    text = "Patta Passbook with survey numbers and land details."
    assert classify_document_category(text) == "Land/Property Document"

def test_aadhaar_category():
    text = "Unique Identification Authority of India Aadhaar Card with ID number."
    assert classify_document_category(text) == "Identity Document"

def test_pan_category():
    text = "Permanent Account Number PAN card issued by Income Tax Department."
    assert classify_document_category(text) == "Financial Document"

def test_birth_certificate_category():
    text = "Birth Certificate issued by municipal authority with date of birth."
    assert classify_document_category(text) == "Certificate"

def test_court_order_category():
    text = "IN THE HIGH COURT OF MADRAS. Court Order is hereby passed."
    assert classify_document_category(text) == "Legal Document"
