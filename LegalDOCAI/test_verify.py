
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.app.services.ml_service import classifier

def test_verification():
    # Example Sale Deed text
    text = """
    THIS DEED OF SALE is made and executed on this 12th day of February 2026.
    BETWEEN Mr. John Doe, hereinafter referred to as the VENDOR.
    AND Mr. Jane Smith, hereinafter referred to as the VENDEE.
    The Vendor is the absolute owner of the property described in the schedule.
    The Vendor hereby sells and transfers the property to the Vendee for a consideration of Rs. 10,00,000.
    In witness whereof the parties have signed this deed.
    """
    
    print("Testing Sale Deed verification...")
    result = classifier.verify_document_ml(text)
    print(f"Result: {result}")
    
    # Example short text
    text_short = "This is a random document."
    print("\nTesting short text verification...")
    result_short = classifier.verify_document_ml(text_short)
    print(f"Result: {result_short}")

if __name__ == "__main__":
    test_verification()
