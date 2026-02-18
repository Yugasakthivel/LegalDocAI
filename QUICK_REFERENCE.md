# Quick Reference

Last updated: 2026-02-15

## Core Commands

```powershell
# Backend (FastAPI)
cd LegalDOCAI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
set DISABLE_MONGO=1
python main.py   # http://127.0.0.1:8000

# Frontend (Vite React)
cd ..\LegalDoc-FrontEnd
npm install
npm run local    # http://127.0.0.1:5173
```

## Useful Checks

```powershell
# Backend health
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing

# Frontend health (headers)
curl.exe -I http://localhost:5173/

# Ports
netstat -ano | findstr :8000
netstat -ano | findstr :5173

# Process for PID
Get-Process -Id <PID> | Select-Object Id,ProcessName,Path

# Verify frontend process command line
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object ProcessId,CommandLine
```

## Important Files

- `LegalDOCAI/backend/app/core/config.py`
- `LegalDOCAI/backend/app/routes/__init__.py`
- `LegalDOCAI/requirements.txt`
- `LegalDoc-FrontEnd/vite.config.js`
- `LegalDoc-FrontEnd/package.json`
- `UNIFIED_DASHBOARD_ARCHITECTURE.md`
- `IMPLEMENTATION_SUMMARY.md`

## Troubleshooting

1. `validate.ps1` reports missing backend deps

```powershell
cd LegalDOCAI
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Frontend shows 404 on `:5173`

- Another Node process may be using that port.
- Confirm PID and command line.
- If it is not `LegalDoc-FrontEnd\\node_modules\\vite\\bin\\vite.js`, run `npm run local` in `LegalDoc-FrontEnd`.

3. Backend unavailable

```powershell
cd LegalDOCAI
python main.py
```

## Env Templates

Backend `.env` (create in `LegalDOCAI`):

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
MONGO_URI=mongodb://localhost:27017
DB_NAME=LegalDocAI_DB
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
UPLOAD_FOLDER=uploads
ACCESS_TOKEN_EXPIRE_MINUTES=60
SECRET_KEY=change-me
ALGORITHM=HS256
```

Frontend `.env` (create in `LegalDoc-FrontEnd`):

```dotenv
VITE_BACKEND_URL=http://127.0.0.1:8000
```

## Key Endpoints

- `GET /api/health` – health check
- `POST /api/modules/ingest` – upload/ingest file
- `POST /api/analysis/advanced` – full analysis
- `POST /api/search/rag?doc_id=...&question=...` – Q&A
- `GET /api/report/html?doc_id=...` – HTML report
- `GET /api/document/history` – history list
