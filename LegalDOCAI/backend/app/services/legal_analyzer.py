from backend.app.services.document_classifier import classify_document
from backend.app.services.entity_extractor import extract_entities

def analyze_document(text):
    classification = classify_document(text or "")
    entities = extract_entities(text or "")
    return {
        "category": classification["category"],
        "document_type": classification["document_type"],
        "entities": entities
    }
