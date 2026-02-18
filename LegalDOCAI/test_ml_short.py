
import sys
import os
from backend.app.services.ml_service import classifier

def test_short_rule_match():
    # Text shorter than 100 chars but containing Sale Deed keywords
    text = "sale deed vendor vendee schedule property consideration"
    print(f"Testing short text (length {len(text)}): '{text}'")
    
    res = classifier.verify_document_ml(text)
    print(f"Verification Result: {res}")
    
    if res['predicted_category'] == 'Sale Deed' and res['marker'] != 'Rejected':
        print("SUCCESS: Short text rule-match handled correctly.")
    else:
        print("FAILURE: Short text rule-match still being rejected.")

if __name__ == "__main__":
    test_short_rule_match()
