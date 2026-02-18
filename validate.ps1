# ============================================
# LegalDocAI - Validation Script
# ============================================
# This script validates the setup and checks for common issues.

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LegalDocAI - Setup Validation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$issues = @()
$warnings = @()

# ============ SYSTEM CHECKS ============
Write-Host "[1/5] Checking System Requirements..." -ForegroundColor Yellow
Write-Host ""

try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(\d+)") {
        $minorVersion = [int]$Matches[1]
        if ($minorVersion -ge 11) {
            Write-Host "  [OK] Python: $pythonVersion" -ForegroundColor Green
        }
        else {
            Write-Host "  [WARN] Python: $pythonVersion (Recommended: 3.11+)" -ForegroundColor Yellow
            $warnings += "Python version is older than recommended"
        }
    }
}
catch {
    Write-Host "  [ERR] Python not found" -ForegroundColor Red
    $issues += "Python is not installed or not in PATH"
}

try {
    $nodeVersion = node --version 2>&1
    if ($nodeVersion -match "v(\d+)") {
        $majorVersion = [int]$Matches[1]
        if ($majorVersion -ge 18) {
            Write-Host "  [OK] Node.js: $nodeVersion" -ForegroundColor Green
        }
        else {
            Write-Host "  [WARN] Node.js: $nodeVersion (Recommended: v18+)" -ForegroundColor Yellow
            $warnings += "Node.js version is older than recommended"
        }
    }
}
catch {
    Write-Host "  [ERR] Node.js not found" -ForegroundColor Red
    $issues += "Node.js is not installed or not in PATH"
}

try {
    $mongoStatus = Get-Service -Name MongoDB -ErrorAction SilentlyContinue
    if ($null -ne $mongoStatus) {
        if ($mongoStatus.Status -eq "Running") {
            Write-Host "  [OK] MongoDB service is running" -ForegroundColor Green
        }
        else {
            Write-Host "  [WARN] MongoDB service is stopped" -ForegroundColor Yellow
            $warnings += "MongoDB service is not running (app will use JSON fallback if implemented)"
        }
    }
    else {
        Write-Host "  [WARN] MongoDB service not found" -ForegroundColor Yellow
        $warnings += "MongoDB not installed as a service"
    }
}
catch {
    Write-Host "  [WARN] Unable to check MongoDB status" -ForegroundColor Yellow
}

Write-Host ""

# ============ BACKEND CHECKS ============
Write-Host "[2/5] Checking Backend Configuration..." -ForegroundColor Yellow
Write-Host ""

Set-Location -Path "LegalDOCAI"

if (Test-Path "venv") {
    Write-Host "  [OK] Python virtual environment exists" -ForegroundColor Green
}
else {
    Write-Host "  [ERR] Virtual environment missing" -ForegroundColor Red
    $issues += "Backend virtual environment not found"
}

if (Test-Path ".env") {
    Write-Host "  [OK] .env file exists" -ForegroundColor Green

    $backendEnv = Get-Content ".env" -Raw
    if ($backendEnv -match "(?m)^OPENAI_API_KEY=.+") {
        Write-Host "    [OK] OPENAI_API_KEY is set" -ForegroundColor Green
    }
    else {
        Write-Host "    [WARN] OPENAI_API_KEY not set" -ForegroundColor Yellow
        $warnings += "OPENAI_API_KEY is missing (RAG/chat features may fail)"
    }

    if ($backendEnv -match "(?m)^TESSERACT_CMD=.+") {
        Write-Host "    [OK] TESSERACT_CMD is set" -ForegroundColor Green
    }
    else {
        Write-Host "    [WARN] TESSERACT_CMD not set" -ForegroundColor Yellow
        $warnings += "Tesseract OCR path not configured"
    }
}
else {
    Write-Host "  [ERR] .env file missing" -ForegroundColor Red
    $issues += "Backend .env file not found"
}

if (Test-Path "venv\Lib\site-packages\fastapi") {
    Write-Host "  [OK] Python dependencies appear installed" -ForegroundColor Green
}
else {
    Write-Host "  [ERR] Python dependencies missing" -ForegroundColor Red
    $issues += "Backend dependencies missing (run: pip install -r requirements.txt)"
}

$requiredDirs = @("uploads", "chroma")
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "  [OK] Directory exists: $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  [WARN] Directory missing: $dir (will be auto-created by runtime/scripts)" -ForegroundColor Yellow
    }
}

Set-Location -Path ".."
Write-Host ""

# ============ FRONTEND CHECKS ============
Write-Host "[3/5] Checking Frontend Configuration..." -ForegroundColor Yellow
Write-Host ""

Set-Location -Path "LegalDoc-FrontEnd"

if (Test-Path "node_modules") {
    Write-Host "  [OK] Node dependencies installed" -ForegroundColor Green
}
else {
    Write-Host "  [ERR] Node dependencies missing" -ForegroundColor Red
    $issues += "Frontend dependencies missing (run: npm install)"
}

if (Test-Path ".env") {
    Write-Host "  [OK] .env file exists" -ForegroundColor Green
    $frontendEnv = Get-Content ".env" -Raw
    if ($frontendEnv -match "(?m)^VITE_BACKEND_URL=.+") {
        Write-Host "    [OK] VITE_BACKEND_URL is set" -ForegroundColor Green
    }
    else {
        Write-Host "    [WARN] VITE_BACKEND_URL not set" -ForegroundColor Yellow
        $warnings += "Frontend backend URL not set"
    }
}
else {
    Write-Host "  [ERR] .env file missing" -ForegroundColor Red
    $issues += "Frontend .env file not found"
}

Set-Location -Path ".."
Write-Host ""

# ============ NETWORK CHECKS ============
Write-Host "[4/5] Checking Network Ports..." -ForegroundColor Yellow
Write-Host ""

$ports = @(
    @{ Port = 8000; Name = "Backend (FastAPI)" },
    @{ Port = 5173; Name = "Frontend (Vite)" }
)

foreach ($portInfo in $ports) {
    $connection = Test-NetConnection -ComputerName localhost -Port $portInfo.Port -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        Write-Host "  [WARN] Port $($portInfo.Port) is in use ($($portInfo.Name))" -ForegroundColor Yellow
        $warnings += "Port $($portInfo.Port) is already in use"
    }
    else {
        Write-Host "  [OK] Port $($portInfo.Port) is available ($($portInfo.Name))" -ForegroundColor Green
    }
}

Write-Host ""

# ============ FILE STRUCTURE CHECKS ============
Write-Host "[5/5] Checking Project Structure..." -ForegroundColor Yellow
Write-Host ""

$criticalPaths = @(
    "LegalDOCAI\main.py",
    "LegalDOCAI\backend\app\routes\modules.py",
    "LegalDoc-FrontEnd\src\App.jsx",
    "LegalDoc-FrontEnd\src\context\DocumentContext.jsx",
    "LegalDoc-FrontEnd\package.json"
)

foreach ($path in $criticalPaths) {
    if (Test-Path $path) {
        Write-Host "  [OK] $path" -ForegroundColor Green
    }
    else {
        Write-Host "  [ERR] $path" -ForegroundColor Red
        $issues += "Critical file missing: $path"
    }
}

Write-Host ""

# ============ SUMMARY ============
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Validation Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($issues.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "  [OK] All checks passed. Ready to run." -ForegroundColor Green
}
else {
    if ($issues.Count -gt 0) {
        Write-Host "  [ERR] Critical issues: $($issues.Count)" -ForegroundColor Red
        foreach ($issue in $issues) {
            Write-Host "    - $issue" -ForegroundColor Red
        }
        Write-Host ""
    }

    if ($warnings.Count -gt 0) {
        Write-Host "  [WARN] Warnings: $($warnings.Count)" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "    - $warning" -ForegroundColor Yellow
        }
        Write-Host ""
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($issues.Count -gt 0) {
    Write-Host "Fix critical issues and re-run .\\setup.ps1 as needed." -ForegroundColor Yellow
    exit 1
}
else {
    Write-Host "Run .\\start.ps1 to launch the application." -ForegroundColor Green
    exit 0
}
