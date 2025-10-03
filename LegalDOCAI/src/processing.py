# backend/app/processing.py

from typing import List

# Example: simple document processing (can add NLP later)
def process_document(text: str) -> dict:
    """
    Process a single document.
    Returns a dictionary with processed content.
    """
    if not text:
        raise ValueError("Document text cannot be empty")
    
    # Example: basic processing
    processed = {
        "original": text,
        "length": len(text),
        "summary": text[:100] + "..." if len(text) > 100 else text
    }
    return processed


def process_documents_bulk(texts: List[str]) -> List[dict]:
    """
    Process multiple documents.
    """
    results = []
    for text in texts:
        try:
            results.append(process_document(text))
        except ValueError:
            results.append({"error": "Empty document"})
    return results
