# ============================================
# LegalDocAI - Automated Setup Script
# ============================================
# This script sets up both backend and frontend

param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LegalDocAI - Automated Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============ BACKEND SETUP ============
if (-not $SkipBackend) {
    Write-Host "[1/2] Setting up Backend (FastAPI + Python)..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Python not found. Please install Python 3.11+." -ForegroundColor Red
        exit 1
    }
    
    # Navigate to backend
    Set-Location -Path "LegalDOCAI"
    
    # Create virtual environment if missing
    if (-not (Test-Path "venv")) {
        Write-Host "  → Creating Python virtual environment..." -ForegroundColor Cyan
        python -m venv venv
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Virtual environment exists" -ForegroundColor Green
    }
    
    # Activate virtual environment
    Write-Host "  → Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "  → Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip --quiet
    
    # Install dependencies
    Write-Host "  → Installing Python dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt --quiet
    
    # Download spaCy model
    Write-Host "  → Downloading spaCy English model..." -ForegroundColor Cyan
    python -m spacy download en_core_web_sm --quiet
    
    # Check .env file
    if (-not (Test-Path ".env")) {
        Write-Host "  ⚠ .env file not found. Copying from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "  → Please update .env with your API keys!" -ForegroundColor Yellow
    } else {
        Write-Host "  ✓ .env file exists" -ForegroundColor Green
    }
    
    # Create required directories
    $dirs = @("uploads", "uploads/processed", "chroma", "data/datasets")
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Write-Host "  ✓ Required directories created" -ForegroundColor Green
    
    Set-Location -Path ".."
    Write-Host ""
    Write-Host "  ✅ Backend setup complete!" -ForegroundColor Green
    Write-Host ""
}

# ============ FRONTEND SETUP ============
if (-not $SkipFrontend) {
    Write-Host "[2/2] Setting up Frontend (React + Vite)..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check Node.js
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "  ✓ Node.js found: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Node.js not found. Please install Node.js 18+." -ForegroundColor Red
        exit 1
    }
    
    # Navigate to frontend
    Set-Location -Path "LegalDoc-FrontEnd"
    
    # Install dependencies
    if ($Force -or -not (Test-Path "node_modules")) {
        Write-Host "  → Installing Node.js dependencies..." -ForegroundColor Cyan
        npm install
        Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  ✓ node_modules exists (use -Force to reinstall)" -ForegroundColor Green
    }
    
    # Check .env file
    if (-not (Test-Path ".env")) {
        Write-Host "  ⚠ .env file not found. Creating default .env..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "  ✓ .env file created" -ForegroundColor Green
    } else {
        Write-Host "  ✓ .env file exists" -ForegroundColor Green
    }
    
    Set-Location -Path ".."
    Write-Host ""
    Write-Host "  ✅ Frontend setup complete!" -ForegroundColor Green
    Write-Host ""
}

# ============ SUMMARY ============
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Update .env files with your API keys" -ForegroundColor White
Write-Host "     - Backend: LegalDOCAI\.env" -ForegroundColor Gray
Write-Host "     - Frontend: LegalDoc-FrontEnd\.env" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Start the application:" -ForegroundColor White
Write-Host "     .\start.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Validate the setup:" -ForegroundColor White
Write-Host "     .\validate.ps1" -ForegroundColor Cyan
Write-Host ""
