import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse, AnalysisResult, EntityData, PartyInfo, ClauseInfo, RiskAnalysis
from app.utils.validators import validate_file, sanitize_filename
from backend.app.core.config import UPLOAD_FOLDER
from backend.app.services import ocr_service
from backend.app.services.storage_service import create_initial_document, save_document
from backend.app.routes.pipeline import run_full_pipeline
from typing import List
import uuid

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_type = validate_file(file)
    sanitized_name = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, sanitized_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    try:
        eff_lang = ocr_service.resolve_ocr_lang_for_file(file_path, file_type, "auto")
        pages: List[str] = ocr_service.extract_text_by_filetype(file_path, file_type, ocr_lang=eff_lang)
        extracted_text = "\n".join(pages)
    except Exception as e:
        try:
            os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
    try:
        doc_id = str(uuid.uuid4())
        create_initial_document(doc_id, file.filename, file_type, extracted_text, pages=pages)
        try:
            save_document(doc_id, {
                "file_info": {
                    "filename": file.filename,
                    "type": file_type,
                    "pages": len(pages or []),
                    "ocr_lang": eff_lang
                },
                "analytics": {
                    "pages": len(pages or [])
                }
            })
        except Exception:
            pass
        pipeline = await run_full_pipeline(doc_id)
        analytics = pipeline.get("analytics") or {}
        results = pipeline.get("results_summary") or {}
        parties = [PartyInfo(name=n, role=None) for n in results.get("names", [])]
        if results.get("organizations"):
            parties += [PartyInfo(name=o, role=None) for o in results.get("organizations", [])]
        entities = EntityData(
            parties=parties,
            dates=results.get("dates", []),
            amounts=[],
            properties=[],
            survey_numbers=[],
            case_numbers=[],
            registration_numbers=[],
            courts=[]
        )
        clauses_raw = (analytics.get("clause_risk") or {}).get("clauses", [])
        clauses = [ClauseInfo(category="other", content=(c.get("clause") or "")) for c in clauses_raw]
        key_terms = list((analytics.get("keyword_frequency") or {}).keys())
        risk = RiskAnalysis(
            level=(analytics.get("clause_risk") or {}).get("overall_risk", "UNKNOWN") or "UNKNOWN",
            details=[c.get("reason","") for c in clauses_raw if c.get("reason")]
        )
        analysis = AnalysisResult(
            document_type=str(analytics.get("document_type") or "Unknown Legal Document"),
            summary_simple=str(analytics.get("summary") or ""),
            entities=entities,
            clauses=clauses,
            key_terms=key_terms,
            risk_analysis=risk,
            compliance_flags=(analytics.get("mandatory_fields") or {}).get("missing", []),
            confidence_score=float((analytics.get("legal_document_confidence") or 0)/100.0)
        )
    except Exception as e:
        try:
            os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to analyze document: {str(e)}")
    try:
        os.remove(file_path)
    except Exception:
        pass
    return UploadResponse(
        status="success",
        filename=file.filename,
        file_type=file_type,
        extracted_text=extracted_text,
        analysis=analysis,
        disclaimer="This analysis is for educational purposes only and does not constitute legal advice."
    )
