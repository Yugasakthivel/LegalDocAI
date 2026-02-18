from backend.app.services.legal_analyzer import analyze_document

def test_sale_deed_basic():
    text = "Sale Deed executed between vendor and purchaser for consideration ₹15,00,000. S.No.142/3 dated 12/05/2022."
    res = analyze_document(text)
    assert res["category"] in ["Land/Legal Document", "Legal Document", "Land Document"]
    assert res["document_type"] == "Sale Deed"
    assert "12/05/2022" in res["entities"]["dates"]
    assert "₹15,00,000" in res["entities"]["amounts"]
    assert any("S.No.142/3" in s for s in res["entities"]["survey_numbers"])
