from typing import List, Dict, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.config import settings
from app.models.schemas import AnalysisResult

_client: Optional[MongoClient] = None
_db = None
_collection = None

def init_database():
    """Initialize MongoDB connection"""
    global _client, _db, _collection
    
    try:
        _client = MongoClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )
        
        _client.admin.command('ping')
        
        _db = _client[settings.MONGODB_DB_NAME]
        _collection = _db["documents"]
        
        print(f"MongoDB connected: {settings.MONGODB_DB_NAME}")
        return True
    
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection failed: {e}")
        print("Database features will be disabled. Install and run MongoDB to enable.")
        return False
    
    except Exception as e:
        print(f"Unexpected database error: {e}")
        return False

def is_database_connected() -> bool:
    """Check if database is connected"""
    return _collection is not None

def save_document_analysis(
    filename: str,
    file_type: str,
    extracted_text: str,
    analysis: AnalysisResult
) -> Optional[str]:
    """
    Save document analysis to MongoDB
    
    Args:
        filename: Original filename
        file_type: File extension
        extracted_text: Extracted text content
        analysis: Analysis results
    
    Returns:
        Document ID if saved, None if database unavailable
    """
    if not is_database_connected():
        return None
    
    try:
        document = {
            "filename": filename,
            "file_type": file_type,
            "extracted_text": extracted_text[:5000],
            "analysis": analysis.model_dump(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = _collection.insert_one(document)
        return str(result.inserted_id)
    
    except Exception as e:
        print(f"Error saving document: {e}")
        return None

def get_document_history(limit: int = 50) -> List[Dict]:
    """
    Retrieve document analysis history
    
    Args:
        limit: Maximum number of documents to return
    
    Returns:
        List of document records
    """
    if not is_database_connected():
        return []
    
    try:
        documents = list(_collection.find(
            {},
            {"_id": 0}
        ).sort("created_at", -1).limit(limit))
        
        return documents
    
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []

def get_analytics_summary() -> Dict:
    """
    Get analytics summary across all documents
    
    Returns:
        Summary statistics
    """
    if not is_database_connected():
        return {
            "total_documents": 0,
            "document_types": {},
            "risk_levels": {},
            "database_status": "disconnected"
        }
    
    try:
        total_docs = _collection.count_documents({})
        
        pipeline = [
            {
                "$group": {
                    "_id": "$analysis.document_type",
                    "count": {"$sum": 1}
                }
            }
        ]
        doc_types = {item["_id"]: item["count"] for item in _collection.aggregate(pipeline)}
        
        pipeline_risk = [
            {
                "$group": {
                    "_id": "$analysis.risk_analysis.level",
                    "count": {"$sum": 1}
                }
            }
        ]
        risk_levels = {item["_id"]: item["count"] for item in _collection.aggregate(pipeline_risk)}
        
        return {
            "total_documents": total_docs,
            "document_types": doc_types,
            "risk_levels": risk_levels,
            "database_status": "connected"
        }
    
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return {
            "total_documents": 0,
            "document_types": {},
            "risk_levels": {},
            "database_status": "error"
        }

init_database()
