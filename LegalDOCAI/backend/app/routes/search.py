from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.app.vectorstore import search, fetch_document_by_id, add_document, get_chroma_collection, embed_texts
from backend.app.services import ocr_service
from backend.app.services.offline_rag_service import generate_answer_offline
from backend.app.services.storage_service import save_document
from backend.app.core.deps import get_current_user, is_openai_ready
import uuid
import re

router = APIRouter()


def _openai_configured() -> bool:
    return is_openai_ready()


def _extract_simple_field(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            return m.group(1).strip(" .,:;-")
    return None


def _local_qa_fallback(doc: Dict[str, Any], question: str) -> str:
    q = (question or "").strip().lower()
    analytics = doc.get("analytics") or {}
    filename = str(doc.get("filename") or "").strip()
    doc_type = str(analytics.get("document_type") or "").strip()
    text = (doc.get("combined_text") or doc.get("text") or "")[:15000]

    # Metadata-first answers: useful even when OpenAI is unavailable.
    if any(k in q for k in ["name of doc", "document name", "doc name", "file name", "filename"]):
        if filename:
            return f"Document name: {filename}"
        return "Document name is not available."

    if any(k in q for k in ["what is this document", "document type", "type of doc", "type"]):
        if doc_type:
            return f"Document type: {doc_type}"
        return "Document type is not available."

    if any(k in q for k in ["confidence", "ai confidence"]):
        conf = analytics.get("ai_confidence")
        if conf is not None and str(conf).strip() != "":
            return f"AI confidence: {conf}%"
        return "AI confidence is not available."

    if any(k in q for k in ["legal status", "status"]):
        status = analytics.get("legal_status")
        if status:
            return f"Legal status: {status}"
        return "Legal status is not available."

    if any(k in q for k in ["name", "holder name", "person name"]):
        val = _extract_simple_field(
            text,
            [
                r"\bname\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{1,80})",
                r"\bholder\s*name\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{1,80})",
            ],
        )
        if val:
            return f"Name found: {val}"
        return "Name was not clearly found in this document text."

    return "AI answering is not configured on the server. Ask about document name, type, legal status, confidence, or name field."

@router.get("/search/")
def search_documents(q: str = Query(..., min_length=1), k: int = 5, user: dict = Depends(get_current_user)):
    """
    Perform semantic search on documents.
    q = query (required), k = number of top results (default=5).
    """
    if not _openai_configured():
        return JSONResponse(
            {"query": q, "results": [], "error": "OpenAI not configured"},
            status_code=503,
        )

    hits: List[Dict] = search(q, top_k=k)
    expanded: List[Dict] = []

    for h in hits:
        doc: Dict = fetch_document_by_id(h.get("doc_id") or "")
        expanded.append({
            "doc_id": h.get("doc_id"),
            "score": h.get("score"),
            "filename": doc.get("filename"),
            "text_preview": (doc.get("text") or doc.get("combined_text") or "")[:400],
        })

    return {"query": q, "results": expanded}

@router.post("/rag/upsert")
async def rag_upsert(text: str = Form(None), doc_id: str = Form(None), source: str = Form(None), user: dict = Depends(get_current_user)):
    try:
        print(f"[DEBUG] rag_upsert: doc_id={doc_id}, source={source}")
        payload_text = text or ""
        if not payload_text and doc_id:
            d = fetch_document_by_id(doc_id)
            payload_text = d.get("combined_text", "") if d else ""
        if not payload_text:
            print("[DEBUG] rag_upsert: No text found")
            return JSONResponse({"indexed": False, "error": "No text to upsert"}, status_code=400)

        print(f"[DEBUG] rag_upsert: adding document, text_len={len(payload_text)}")
        new_id = add_document({"doc_id": doc_id, "filename": (source or "")[:200], "combined_text": payload_text})
        print(f"[DEBUG] rag_upsert: success, new_id={new_id}")
        return {"id": new_id, "doc_id": new_id, "size": len(payload_text), "indexed": True}
    except Exception as e:
        print(f"[DEBUG] rag_upsert error: {e}")
        # Always succeed for frontend UX, but mark as not indexed
        return JSONResponse(
            {
                "id": doc_id or "",
                "doc_id": doc_id or "",
                "size": len(text or ""),
                "indexed": False,
                "warning": "Indexing failed; stored document only.",
                "detail": str(e),
            },
            status_code=200,
        )

@router.post("/rag")
async def rag_query_endpoint(doc_id: str = Query(...), question: str = Query(...), user: dict = Depends(get_current_user)):
    """Simple RAG endpoint for frontend chat"""
    # 1. Retrieve document text
    doc = fetch_document_by_id(doc_id)
    if not doc:
        return {"answer": "Document not found."}

    # Local deterministic fallback when OpenAI is unavailable.
    if not _openai_configured():
        text = doc.get("combined_text") or doc.get("text") or ""
        data = generate_answer_offline(question, text or "", doc.get("analytics") or {})
        return {"answer": data.get("answer") or "", "source": data.get("source") or "offline_rag", "confidence": data.get("confidence")}
    
    # Determine detected type for gating
    analytics = doc.get("analytics") or {}
    detected_type = str(analytics.get("document_type") or "").lower()
    q_l = (question or "").strip().lower()
    # Type groups
    gov_id_types = {"pan card","e-pan","aadhaar card","aadhaar pvc card","e-aadhaar","passport","indian passport","driving licence","driving license","learner licence","learner license","voter id","e-voter id","epic","e-epic","ration card","e-ration card","oci card","pio card","id card","identity card","employee id"}
    contract_types = {"lease deed","rental agreement","employment contract","partnership deed","corporate agreement","legal agreement","contract"}
    invoice_types = {"invoice"}
    bank_types = {"bank statement"}
    property_types = {"patta","tslr","encumbrance certificate","sale deed","gift deed","land property"}
    court_types = {"court judgment","court order","legal notice","affidavit"}
    certificate_types = {"birth certificate","death certificate","transfer certificate","bonafide certificate","salary certificate","gst certificate"}
    other_types = {"other indian legal document","resume"}
    # Allowed topic keywords per group
    identity_allowed = any(k in q_l for k in ["name","holder","person","authority","issuing authority","issuer","issue date","issued on","expiry","valid till","validity","dob","date of birth","address","father","mother","pan","aadhaar","passport","license","licence","dl","id number","card number","number","uidai"])
    contract_allowed = any(k in q_l for k in ["party","parties","termination","rent","payment","liability","confidentiality","obligation","obligations","rights","jurisdiction","governing law","indemnity","advance","penalty","fees"])
    invoice_allowed = any(k in q_l for k in ["invoice","gst","gstin","bill","billing","line item","items","subtotal","total","amount due","tax","igst","cgst","sgst","invoice number"])
    bank_allowed = any(k in q_l for k in ["account","statement","transaction","transactions","balance","debit","credit","ifsc","account number"])
    property_allowed = any(k in q_l for k in ["survey number","survey","property","plot","title","ownership","encumbrance","tslr","patta","sale deed","gift deed","registration","registration number","land","extent"])
    court_allowed = any(k in q_l for k in ["case","case number","court","judgment","order","legal notice","affidavit","party","petitioner","respondent","date"])
    cert_allowed = any(k in q_l for k in ["certificate","issued on","issue date","authority","reference","id","number","name","dob","date of birth","father","mother","address","salary","gst"])
    # Gate by type group
    t = detected_type
    if any(tt in t for tt in gov_id_types):
        if not identity_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in contract_types):
        # contracts allowed as is
        pass
    elif any(tt in t for tt in invoice_types):
        if not invoice_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in bank_types):
        if not bank_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in property_types):
        if not property_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in court_types):
        if not court_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in certificate_types):
        if not cert_allowed:
            return {"answer": "The document does not contain this information."}
    elif any(tt in t for tt in other_types):
        # Limit to generic identity/metadata
        generic_allowed = identity_allowed or any(k in q_l for k in ["date","number","reference","issuer","authority","name","address","party","organization"])
        if not generic_allowed:
            return {"answer": "The document does not contain this information."}

    text = doc.get("combined_text") or doc.get("text") or ""
    if not text:
        return {"answer": "Document has no text content."}

    # 2. Simple 'search' or 'ask' logic
    # Ideally, we chunk and vector search, but for MVP we can send context to LLM if small enough
    # or use the existing vectorstore functions.
    
    # Check if we have a vector index for this doc? 
    # For now, let's use a direct LLM call with context (since we have the text).
    # This is a fallback if vector store isn't fully set up for this doc yet.
    
    from backend.app.services.analysis_service import answer_with_fallback, detect_language, translate_to_english_async, correct_ocr_errors_async
    
    # Truncate text to fit context if needed (approx 10k chars)
    # Prefer translated/corrected working text for better extraction
    try:
        lang = detect_language(text[:2000])
        if lang != "English":
            working = await translate_to_english_async(text[:6000], lang)
        else:
            working = await correct_ocr_errors_async(text[:6000])
    except Exception:
        working = text[:6000]
    context = (working or text)[:15000]
    ans = answer_with_fallback(question, context)
    return {"answer": ans}
