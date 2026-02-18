# Quick Start

Last updated: 2026-02-15

## Fastest Path

```powershell
cd d:\Legal-mohan\Legaldoc-new
.\setup.ps1
.\validate.ps1
.\start.ps1
```

## Verify Backend

```powershell
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
```

Expected status code: `200`

## Verify Frontend

Open `http://localhost:5173` in browser.

If you see 404 or wrong app:
1. Check process on port 5173.
2. Start frontend from this repo:

```powershell
cd LegalDoc-FrontEnd
npm run local
```

## Minimum Env Setup

Backend (`LegalDOCAI/.env`):

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=LegalDocAI_DB
OPENAI_API_KEY=...
TESSERACT_CMD=...
```

Frontend (`LegalDoc-FrontEnd/.env`):

```env
VITE_BACKEND_URL=http://localhost:8000
```
