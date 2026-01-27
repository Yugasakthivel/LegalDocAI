from fastapi import APIRouter
from app.services.database_service import get_document_history, get_analytics_summary

router = APIRouter()

@router.get("/history")
def document_history(limit: int = 50):
    """
    Get document analysis history
    
    Args:
        limit: Maximum number of documents to return (default: 50)
    
    Returns:
        List of analyzed documents
    """
    documents = get_document_history(limit=limit)
    
    return {
        "status": "success",
        "count": len(documents),
        "documents": documents
    }

@router.get("/analytics")
def analytics_summary():
    """
    Get analytics summary across all analyzed documents
    
    Returns:
        Summary statistics including document types, risk levels, etc.
    """
    summary = get_analytics_summary()
    
    return {
        "status": "success",
        "summary": summary
    }
