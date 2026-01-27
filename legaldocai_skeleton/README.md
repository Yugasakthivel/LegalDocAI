# LegalDocAI Backend Skeleton

**Backend skeleton for Indian Legal Document Analysis System**

This is TASK 1 - a clean, production-ready FastAPI skeleton with placeholder services for OCR and LLM integration.

## Features

- ✅ File upload endpoint (PDF, JPG, PNG, DOCX, TXT)
- ✅ File validation and sanitization
- ✅ OCR service stub (ready for Tesseract integration)
- ✅ LLM analysis service stub (ready for OpenAI/Anthropic integration)
- ✅ Structured JSON response format
- ✅ CORS enabled for frontend integration
- ✅ Clean architecture (routes, services, models, utils)

## Supported Document Types (Future)

- Sale Deed
- Lease Deed
- Rental Agreement
- Gift Deed
- Court Judgments & Orders
- Legal Notices
- Affidavits
- Employment Contracts
- Partnership Deeds
- And more Indian legal documents

## Project Structure

```
legaldocai_skeleton/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py           # File upload endpoint
│   │   └── health.py           # Health check
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py      # OCR placeholder
│   │   ├── llm_service.py      # LLM placeholder
│   │   └── file_service.py     # File handling
│   └── utils/
│       ├── __init__.py
│       └── validators.py       # File validation
├── uploads/                    # Upload directory
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults are fine for development).

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

Server will start at: **http://localhost:8000**

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "LegalDocAI Backend Skeleton",
  "version": "0.1.0"
}
```

### Root

```bash
GET /
```

**Response:**
```json
{
  "service": "LegalDocAI Backend Skeleton",
  "version": "0.1.0",
  "status": "running",
  "disclaimer": "This system is for educational purposes only..."
}
```

### Upload Document

```bash
POST /api/upload
Content-Type: multipart/form-data

Body: file=@document.pdf
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_type": "pdf",
  "extracted_text": "[OCR PLACEHOLDER] Text extraction from PDF...",
  "analysis": {
    "document_type": "Unknown (LLM integration pending)",
    "summary_simple": "This document appears to be a legal document...",
    "entities": {
      "parties": ["[Placeholder] Party identification pending"],
      "dates": [],
      "amounts": [],
      "properties": [],
      "case_numbers": [],
      "courts": []
    },
    "clauses": ["[Placeholder] Ownership clause detection pending", ...],
    "key_terms": [],
    "risk_analysis": {
      "level": "UNKNOWN",
      "details": ["Risk assessment requires LLM integration", ...]
    },
    "compliance_flags": [],
    "confidence_score": 0.0
  },
  "disclaimer": "This analysis is for educational purposes only..."
}
```

## Testing

### Manual Testing with cURL

**Test health endpoint:**
```bash
curl http://localhost:8000/health
```

**Upload a file:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@sample.pdf"
```

### Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Upload file
files = {"file": open("sample.pdf", "rb")}
response = requests.post("http://localhost:8000/api/upload", files=files)
print(response.json())
```

### Interactive API Docs

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Next Steps (Future Tasks)

### TASK 2: OCR Integration
- Integrate Tesseract OCR for scanned documents
- Add PyMuPDF for PDF text extraction
- Add python-docx for Word document parsing
- Support Hindi/regional languages

### TASK 3: LLM Integration
- Integrate OpenAI GPT-4 or Anthropic Claude
- Create prompt templates for Indian legal documents
- Implement document classification
- Add entity extraction (NER)
- Add clause detection and categorization

### TASK 4: Database Integration
- Add MongoDB or PostgreSQL
- Store document analysis history
- Implement document search API

### TASK 5: Advanced Features
- Risk assessment algorithms
- Compliance checking (Indian acts)
- Legal knowledge base
- Confidence scoring

## Placeholder Services

### OCR Service (`app/services/ocr_service.py`)
Currently returns placeholder text. Ready for:
- Tesseract integration
- Image preprocessing (deskew, denoise)
- Multi-language support

### LLM Service (`app/services/llm_service.py`)
Currently returns structured placeholder analysis. Ready for:
- OpenAI/Anthropic API integration
- Custom prompts for Indian legal context
- Entity extraction and clause detection

## Configuration

Environment variables (`.env`):

- `UPLOAD_FOLDER`: Directory for uploaded files (default: `uploads`)
- `MAX_UPLOAD_SIZE_MB`: Max file size in MB (default: `10`)
- `ALLOWED_EXTENSIONS`: Comma-separated file extensions (default: `pdf,jpg,jpeg,png,docx,txt`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## License

Educational use only. Does not constitute legal advice.

## Disclaimer

⚠️ **This system is for educational and decision-support purposes only. It does not provide legal advice. Always consult a qualified legal professional for legal matters.**
