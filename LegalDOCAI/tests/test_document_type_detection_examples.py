import pytest
from backend.app.services.analysis_service import classify_document_type

def test_resume_classification():
    text = "Resume\nEducation\nExperience\nSkills\nProjects\nSummary"
    assert classify_document_type(text) == "Resume"

def test_contract_classification():
    text = "This contract agreement between parties sets terms and conditions."
    assert classify_document_type(text) == "Contract"

def test_invoice_classification():
    text = "Tax Invoice\nInvoice No: INV-123\nGSTIN: 22AAAAA0000A1Z5\nBill To: ACME\nSubtotal\nTotal\nAmount Due"
    assert classify_document_type(text) == "Invoice"

def test_id_card_classification():
    text = "Identity Card\nID No: ABC1234\nDate of Birth: 1990-01-01\nGovernment of India"
    assert classify_document_type(text) == "ID Card"

def test_legal_agreement_classification():
    text = "This agreement is made between the parties and witnesseth the terms herein."
    assert classify_document_type(text) == "Legal Agreement"
