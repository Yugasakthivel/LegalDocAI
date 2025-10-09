# 🚀 LegalDocAI Project Auto Runner
# ---------------------------------
# This script starts both FastAPI backend and React frontend

Write-Host "Starting LegalDocAI Backend & Frontend..." -ForegroundColor Cyan

# ---- BACKEND ----
Write-Host "`nStarting Backend Server (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd D:\Project\LegalDocAI\LegalDOCAI;
     .\venv\Scripts\Activate.ps1;
     python -m spacy download en_core_web_sm;
     uvicorn main:app --reload"
)

# ---- FRONTEND ----
Write-Host "`nStarting Frontend (React/Vite)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd D:\Project\LegalDoc-FrontEnd;
     npm install;
     npm run dev"
)

Write-Host "`n✅ Both Backend and Frontend are launching in separate terminals." -ForegroundColor Green
Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://localhost:5173"
