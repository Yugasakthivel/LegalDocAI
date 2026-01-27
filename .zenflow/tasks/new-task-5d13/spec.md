# Technical Specification - LegalDocAI Backend Skeleton (TASK 1)

## Difficulty Assessment
**Medium** - Involves setting up clean FastAPI architecture with proper separation of concerns, multiple file formats handling, and integration stubs for OCR and LLM services.

---

## Problem Statement

Create a production-ready FastAPI backend skeleton for LegalDocAI that handles Indian legal document uploads, processes them through OCR (placeholder), analyzes them using LLM (placeholder), and returns structured JSON responses.

This is TASK 1 - the foundational skeleton that will support future enhancements for full legal document analysis capabilities.

---

## Scope

### In-Scope
- FastAPI application setup with proper project structure
- File upload endpoint accepting PDF, images (JPG, PNG), Word documents, and plain text
- File type detection and validation
- OCR service stub (placeholder function)
- LLM analysis service stub (placeholder function)
- Structured JSON response format matching Indian legal document requirements
- CORS configuration for future frontend integration
- Basic error handling and validation
- Health check endpoint

### Out-of-Scope (Future Tasks)
- Database integration (MongoDB/PostgreSQL)
- Full OCR implementation with Tesseract
- Full LLM integration with OpenAI/Anthropic
- Advanced clause detection logic
- Compliance checking implementation
- Risk assessment algorithms
- Frontend UI
- Authentication/Authorization
- Production deployment configuration

---

## Existing Codebase Analysis

### Current State
The project has an existing implementation in `LegalDOCAI/`:
- **main.py** (425 lines) - Contains full implementation with MongoDB, OpenAI, NLP analysis
- **config.py** - Configuration management with environment variables
- **backend/app/** - Modular structure with routes, models, utils

### Reusable Components
- Project structure in `backend/app/` can be used as reference
- File processing utilities in `backend/app/processing.py`
- Route organization pattern in `backend/app/routes/`

### Approach
Create a **clean, simplified skeleton** in the root directory that:
1. Follows the same architectural patterns
2. Removes MongoDB, OpenAI, and NLP dependencies for this task
3. Provides clear placeholder functions for future integration
4. Uses minimal dependencies

---

## Technical Context

### Language & Framework
- **Python 3.10+**
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server

### Core Dependencies (Minimal for Skeleton)
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
python-dotenv==1.0.1
pydantic==2.0+
```

### File Structure
```
legaldocai_skeleton/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization
│   ├── config.py               # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py           # File upload endpoint
│   │   └── health.py           # Health check endpoint
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py      # OCR placeholder
│   │   ├── llm_service.py      # LLM analysis placeholder
│   │   └── file_service.py     # File handling utilities
│   └── utils/
│       ├── __init__.py
│       └── validators.py       # File validation utilities
├── uploads/                    # Temporary upload storage
├── .env.example               # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md                   # Setup instructions
```

---

## Implementation Approach

### 1. Application Bootstrap
- Initialize FastAPI app with CORS middleware
- Configure upload folder and file size limits
- Set up basic logging
- Environment-based configuration

### 2. File Upload Flow
```
Client Upload → Validation → Save to Disk → Extract Text (stub) → 
LLM Analysis (stub) → Format Response → Return JSON
```

### 3. Service Layer Stubs

#### OCR Service (Placeholder)
```python
def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    Placeholder: Extract text from uploaded file
    Future: Integrate Tesseract OCR, PyMuPDF, python-docx
    """
    return f"[OCR PLACEHOLDER] Extracted text from {file_type} file"
```

#### LLM Service (Placeholder)
```python
def analyze_legal_document(text: str) -> dict:
    """
    Placeholder: Analyze document using LLM
    Future: Integrate OpenAI/Anthropic for classification, entity extraction, etc.
    """
    return {
        "document_type": "Unknown (placeholder)",
        "summary_simple": "Document analysis pending LLM integration",
        "entities": {},
        "clauses": [],
        "key_terms": [],
        "risk_analysis": {"level": "UNKNOWN", "details": []},
        "compliance_flags": [],
        "confidence_score": 0.0
    }
```

---

## API Specification

### Endpoint: POST /api/upload

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF, JPG, PNG, DOCX, TXT)

**Response (200 OK):**
```json
{
  "status": "success",
  "filename": "sale_deed.pdf",
  "file_type": "pdf",
  "extracted_text": "[OCR PLACEHOLDER] Extracted text from pdf file",
  "analysis": {
    "document_type": "Unknown (placeholder)",
    "summary_simple": "Document analysis pending LLM integration",
    "entities": {
      "parties": [],
      "dates": [],
      "amounts": [],
      "properties": [],
      "case_numbers": [],
      "courts": []
    },
    "clauses": [],
    "key_terms": [],
    "risk_analysis": {
      "level": "UNKNOWN",
      "details": []
    },
    "compliance_flags": [],
    "confidence_score": 0.0
  },
  "disclaimer": "This analysis is for educational purposes only and does not constitute legal advice."
}
```

**Error Response (400/422/500):**
```json
{
  "status": "error",
  "message": "Invalid file type. Supported: PDF, JPG, PNG, DOCX, TXT",
  "detail": "..."
}
```

### Endpoint: GET /health

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "LegalDocAI Backend Skeleton",
  "version": "0.1.0"
}
```

---

## Data Models (Pydantic Schemas)

### UploadResponse
```python
class AnalysisResult(BaseModel):
    document_type: str
    summary_simple: str
    entities: dict
    clauses: list
    key_terms: list
    risk_analysis: dict
    compliance_flags: list
    confidence_score: float

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    extracted_text: str
    analysis: AnalysisResult
    disclaimer: str
```

---

## Configuration

### Environment Variables (.env)
```
UPLOAD_FOLDER=uploads
MAX_UPLOAD_SIZE_MB=10
ALLOWED_EXTENSIONS=pdf,jpg,jpeg,png,docx,txt
LOG_LEVEL=INFO
```

---

## Validation & Error Handling

### File Validation
- Check file extension against allowed list
- Verify file size (max 10MB for skeleton)
- Ensure file is not empty
- Validate file can be read

### Error Handling
- 400: Invalid file type or empty file
- 413: File too large
- 422: Validation error
- 500: Internal server error (with safe error messages)

---

## Verification Approach

### Manual Testing
1. Start server: `uvicorn app.main:app --reload`
2. Test health endpoint: `curl http://localhost:8000/health`
3. Test file upload with different formats:
   - PDF file
   - JPG/PNG image
   - DOCX document
   - TXT file
4. Verify JSON response structure matches specification
5. Test error cases (invalid file type, too large, etc.)

### Automated Testing (Future)
- Unit tests for services
- Integration tests for endpoints
- Load testing for file uploads

---

## Security Considerations

### File Upload Security
- Validate file extensions (whitelist approach)
- Limit file size to prevent DoS
- Save files with sanitized names (UUID-based)
- Store uploads in isolated directory
- Add .gitignore to exclude uploads/

### CORS Configuration
- Start with allow_origins=["*"] for development
- Document need to restrict in production

### Secrets Management
- Use environment variables
- Never commit .env files
- Provide .env.example template

---

## Performance Considerations

### File Handling
- Async file I/O where possible
- Stream large files instead of loading to memory
- Clean up temporary files after processing

### Response Time
- Target: < 2 seconds for skeleton (placeholder responses)
- Future: < 30 seconds for full OCR + LLM analysis

---

## Acceptance Criteria

✅ FastAPI server starts without errors  
✅ Health endpoint returns 200 OK with service information  
✅ Upload endpoint accepts PDF, JPG, PNG, DOCX, TXT files  
✅ Invalid file types return 400 error with clear message  
✅ Response JSON structure matches specification  
✅ Placeholder OCR function returns mock text  
✅ Placeholder LLM function returns structured analysis mock  
✅ Files are saved to uploads/ directory  
✅ CORS is configured for future frontend integration  
✅ README.md includes setup and testing instructions  
✅ .gitignore excludes uploads/ and .env  
✅ Code follows clean architecture principles (separation of concerns)  

---

## Future Integration Points

### OCR Integration (TASK 2)
- Replace `ocr_service.py` stub with Tesseract integration
- Add PyMuPDF for PDF text extraction
- Add python-docx for Word document parsing

### LLM Integration (TASK 3)
- Replace `llm_service.py` stub with OpenAI/Anthropic client
- Implement prompt templates for Indian legal document analysis
- Add confidence scoring logic

### Database Integration (TASK 4)
- Add MongoDB/PostgreSQL models
- Store analysis results
- Implement document history API

### Advanced Analysis (TASK 5+)
- Clause detection algorithms
- Compliance checking against Indian acts
- Risk assessment scoring
- Entity extraction (NER)

---

## Notes

- This skeleton prioritizes **clean architecture** and **extensibility**
- All external integrations (OCR, LLM, DB) are clearly marked as stubs
- Response structure is production-ready and matches full system requirements
- Code should be **self-documenting** with clear function names and docstrings
