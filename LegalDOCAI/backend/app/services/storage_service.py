from backend.app.database import documents_collection
import time

def get_document(doc_id: str):
    return documents_collection.find_one({"doc_id": doc_id})

def save_document(doc_id: str, data: dict):
    # Update existing document with new analysis data
    # We use $set to update only specified fields, preserving existing ones
    documents_collection.update_one(
        {"doc_id": doc_id},
        {"$set": data},
        upsert=True
    )

def create_initial_document(doc_id: str, filename: str, file_type: str, text: str, pages: list = None):
    doc = {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "combined_text": text,
        "upload_timestamp": time.time(),
        "status": "uploaded",
        "analytics": {
            "document_type": "Unknown",
            "verified_marker": "UNVERIFIED",
            "ai_confidence": None,
            "legality_score": 0
        }
    }
    if pages:
        doc["pages"] = pages
        
    documents_collection.insert_one(doc)
    return doc
