# Legal Document Analysis System

## Project Overview

The Legal Document Analysis System helps legal professionals process and understand large volumes of documents. It performs OCR, classification, clause/entity extraction, risk and compliance checks, Q&A (RAG), and reporting, with a modern web UI for upload, analysis, and history.

---

## Key Features

- Upload and manage legal documents (PDF, DOCX, TXT, images)
- OCR text extraction (Tesseract) with multilingual hints
- Document classification (ML) and rule‑based labeling
- Clause and entity extraction, mandatory field checks
- Risk scoring, verification, jurisdiction/compliance checks
- RAG‑style Q&A on document content
- HTML/PDF reports and JSON outputs
- Auth (JWT) and history rehydration (no re‑upload for old docs)

---

## Technology Stack

- Backend: Python FastAPI
- Database: MongoDB (with local JSON fallback via `DISABLE_MONGO=1`)
- OCR: Tesseract (optional EasyOCR fallback)
- NLP/ML: spaCy, scikit‑learn, Transformers/Torch (LLM optional via OpenAI)
- Frontend: React + Vite + Tailwind UI
- Deployment: Docker Compose (Mongo), container‑friendly backend/frontend

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Tesseract OCR installed and available on PATH (Windows example: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- MongoDB (optional for dev; set `DISABLE_MONGO=1` to use local JSON files)

### Installation

1) Backend (FastAPI)

```powershell
cd LegalDOCAI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# .env (create in LegalDOCAI)
# OPENAI_API_KEY=
# OPENAI_MODEL=gpt-4o-mini
# MONGO_URI=mongodb://localhost:27017
# DB_NAME=LegalDocAI_DB
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
# UPLOAD_FOLDER=uploads
# ACCESS_TOKEN_EXPIRE_MINUTES=60
# SECRET_KEY=change-me
# ALGORITHM=HS256

set DISABLE_MONGO=1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

API base: `http://127.0.0.1:8000`  
Docs: `http://127.0.0.1:8000/docs`

2) Frontend (React + Vite)

```powershell
cd LegalDoc-FrontEnd
npm install

# .env (create in LegalDoc-FrontEnd)
# VITE_BACKEND_URL=http://127.0.0.1:8000

npm run local   # http://localhost:5173
```

---

## Project Structure

```
Legal-mohan/Legaldoc-new
├── LegalDOCAI/                 # FastAPI backend
│   ├── main.py                 # App entrypoint
│   └── backend/app/
│       ├── routes/             # API routers (upload, analysis, report, auth, search, pipeline, etc.)
│       ├── services/           # OCR, ML, risk/compliance, verification, storage
│       ├── core/               # Config, dependencies (OpenAI, auth)
│       ├── database.py         # Mongo + JSON fallback
│       └── vectorstore.py      # Embeddings + search
├── LegalDoc-FrontEnd/          # React + Vite frontend
│   └── src/                    # Pages, components, contexts
├── QUICK_REFERENCE.md          # Commands and env templates
├── QUICK_START.md              # Guided setup
└── UNIFIED_DASHBOARD_ARCHITECTURE.md / IMPLEMENTATION_SUMMARY.md
```

---

## API Documentation

Common endpoints (all prefixed with `/api`):

| Method | Endpoint                          | Description                                   |
| ------ | --------------------------------- | --------------------------------------------- |
| GET    | `/api/health`                     | Health check                                  |
| POST   | `/api/auth/register`              | Create account, returns JWT                   |
| POST   | `/api/auth/login`                 | Login, returns JWT                            |
| POST   | `/api/modules/ingest`             | Upload/ingest a document, returns `doc_id`    |
| POST   | `/api/pipeline/full`              | Run full pipeline for a `doc_id`              |
| POST   | `/api/analysis/advanced`          | Analysis from raw text or `doc_id`            |
| POST   | `/api/search/rag?doc_id&question` | Q&A on document content                       |
| GET    | `/api/report/html?doc_id`         | HTML report                                   |
| GET    | `/api/report/pdf?doc_id`          | PDF report (requires PyMuPDF)                 |

### Example: Upload Document

Request:

```bash
curl -X POST http://127.0.0.1:8000/api/modules/ingest \
  -H "Authorization: Bearer <JWT>" \
  -F "file=@path/to/document.pdf"
```

Response:

```json
{
  "doc_id": "uuid-1234",
  "filename": "document.pdf",
  "status": "uploaded"
}
```

Run pipeline:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/full \
  -H "Authorization: Bearer <JWT>" \
  -F "doc_id=uuid-1234"
```

---

## Usage Notes

- Without MongoDB, set `DISABLE_MONGO=1` to use local JSON storage (suitable for dev).
- Set `OPENAI_API_KEY` for LLM‑powered features (optional).
- Ensure `TESSERACT_CMD` is configured for OCR; PDF OCR via PyMuPDF is optional.
- Frontend uses `VITE_BACKEND_URL` or dev proxy for `/api`.

---

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push branch (`git push origin feature-name`)
5. Open a Pull Request with a clear description

Please include tests where applicable and follow the existing code style.
