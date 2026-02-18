# ============================================
# LegalDocAI - Start Script
# ============================================
# This script starts both backend and frontend servers

param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting LegalDocAI Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return -not $connection.TcpTestSucceeded
}

# ============ START BACKEND ============
if (-not $FrontendOnly) {
    Write-Host "[1/2] Starting Backend Server (FastAPI)..." -ForegroundColor Yellow
    
    # Check if port is available
    if (-not (Test-Port -Port $BackendPort)) {
        Write-Host "  ⚠ Port $BackendPort is already in use!" -ForegroundColor Red
        Write-Host "  → Please stop the existing process or use -BackendPort <port>" -ForegroundColor Yellow
        exit 1
    }
    
    # Start backend in new window
    $backendScript = @"
Set-Location 'LegalDOCAI'
& '.\venv\Scripts\Activate.ps1'
Write-Host '========================================' -ForegroundColor Green
Write-Host '  Backend Server Running' -ForegroundColor Green
Write-Host '  URL: http://localhost:$BackendPort' -ForegroundColor Cyan
Write-Host '  API Docs: http://localhost:$BackendPort/docs' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Green
Write-Host ''
python main.py
"@
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript
    Write-Host "  ✓ Backend server starting on http://localhost:$BackendPort" -ForegroundColor Green
    Write-Host "  ✓ API docs available at http://localhost:$BackendPort/docs" -ForegroundColor Green
    Write-Host ""
    
    # Wait for backend to start
    Write-Host "  → Waiting for backend to start..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
}

# ============ START FRONTEND ============
if (-not $BackendOnly) {
    Write-Host "[2/2] Starting Frontend Server (Vite)..." -ForegroundColor Yellow
    
    # Check if port is available
    if (-not (Test-Port -Port $FrontendPort)) {
        Write-Host "  ⚠ Port $FrontendPort is already in use!" -ForegroundColor Red
        Write-Host "  → Please stop the existing process or use -FrontendPort <port>" -ForegroundColor Yellow
        exit 1
    }
    
    # Start frontend in new window
    $frontendScript = @"
Set-Location 'LegalDoc-FrontEnd'
Write-Host '========================================' -ForegroundColor Green
Write-Host '  Frontend Server Running' -ForegroundColor Green
Write-Host '  URL: http://localhost:$FrontendPort' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Green
Write-Host ''
npm run local
"@
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript
    Write-Host "  ✓ Frontend server starting on http://localhost:$FrontendPort" -ForegroundColor Green
    Write-Host ""
}

# ============ SUMMARY ============
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Application Started! 🚀" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Yellow
if (-not $FrontendOnly) {
    Write-Host "  Backend:  http://localhost:$BackendPort" -ForegroundColor Cyan
    Write-Host "  API Docs: http://localhost:$BackendPort/docs" -ForegroundColor Cyan
}
if (-not $BackendOnly) {
    Write-Host "  Frontend: http://localhost:$FrontendPort" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "To stop servers: Close the PowerShell windows" -ForegroundColor Gray
Write-Host ""
