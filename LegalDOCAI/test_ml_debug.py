
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.ml_service import classifier

def test_verification():
    sample_texts = [
        "This is a sale deed for property transfer between vendor and vendee. Consideration is 1000000.",
        "Rental agreement for house. Landlord and tenant sign here.",
        "A short text.",
        "Generic document text without any specific keywords but long enough to be a document."
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Test {i+1} ---")
        result = classifier.verify_document_ml(text)
        print(f"Final Result: {result}")

if __name__ == "__main__":
    test_verification()
