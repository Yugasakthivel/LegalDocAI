# Deployment Guide

Last updated: 2026-02-15

## 1. Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB (optional if backend fallback is enabled)
- Tesseract OCR (optional, needed for OCR features)

## 2. Local Setup (Recommended)

From project root (`Legaldoc-new`):

```powershell
.\setup.ps1
.\validate.ps1
.\start.ps1
```

What this does:
- `setup.ps1`: creates backend venv, installs backend/frontend dependencies, creates env files when missing
- `validate.ps1`: checks system, env files, key folders, and ports
- `start.ps1`: starts backend and frontend in separate PowerShell windows

## 3. Manual Setup

### Backend

```powershell
cd LegalDOCAI
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Copy-Item .env.example .env
python main.py
```

### Frontend

```powershell
cd LegalDoc-FrontEnd
npm install
Copy-Item .env.example .env
npm run local
```

## 4. Required Environment Variables

### `LegalDOCAI/.env`

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=LegalDocAI_DB
OPENAI_API_KEY=...
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### `LegalDoc-FrontEnd/.env`

```env
VITE_BACKEND_URL=http://localhost:8000
```

## 5. Health Verification

Use script:

```powershell
.\validate.ps1
```

Direct checks:

```powershell
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
Invoke-WebRequest http://localhost:8000/docs -UseBasicParsing
curl.exe -I http://localhost:5173/
```

If you need to confirm the frontend process is this repo's app:

```powershell
netstat -ano | findstr :5173
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object ProcessId,CommandLine
```

## 6. Production Notes

- Backend: run FastAPI with `uvicorn` or `gunicorn`+UvicornWorker behind reverse proxy.
- Frontend: build static assets with `npm run build` and serve `dist/` from CDN/Nginx.
- Use HTTPS and keep `.env` values out of source control.

## 7. Common Problems

1. Backend import/dependency errors

```powershell
cd LegalDOCAI
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Frontend cannot call backend

- Check `LegalDoc-FrontEnd/.env` has `VITE_BACKEND_URL=http://localhost:8000`
- Restart Vite after editing `.env`

3. Port conflict

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :5173
Get-Process -Id <PID>
```
