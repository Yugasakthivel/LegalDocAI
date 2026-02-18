# Simple validation script
$ErrorActionPreference = "Continue"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  LegalDocAI - Quick Validation"  -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$issues = 0

# Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVer = python --version 2>&1
    Write-Host "  ✓ $pythonVer" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found" -ForegroundColor Red
    $issues++
}

# Check Node
Write-Host "`n[2/6] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVer = node --version 2>&1
    Write-Host "  ✓ Node.js $nodeVer" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Node.js not found" -ForegroundColor Red
    $issues++
}

# Check Backend
Write-Host "`n[3/6] Checking Backend..." -ForegroundColor Yellow
if (Test-Path "LegalDOCAI\venv") {
    Write-Host "  ✓ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ Virtual environment missing" -ForegroundColor Red
    $issues++
}

if (Test-Path "LegalDOCAI\.env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ .env file missing" -ForegroundColor Red
    $issues++
}

# Check Frontend
Write-Host "`n[4/6] Checking Frontend..." -ForegroundColor Yellow
if (Test-Path "LegalDoc-FrontEnd\node_modules") {
    Write-Host "  ✓ node_modules exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ node_modules missing" -ForegroundColor Red
    $issues++
}

if (Test-Path "LegalDoc-FrontEnd\.env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ .env file missing" -ForegroundColor Red
    $issues++
}

# Check Ports
Write-Host "`n[5/6] Checking Ports..." -ForegroundColor Yellow
$port8000 = Test-NetConnection -ComputerName localhost -Port 8000 -WarningAction SilentlyContinue -InformationLevel Quiet
$port5173 = Test-NetConnection -ComputerName localhost -Port 5173 -WarningAction SilentlyContinue -InformationLevel Quiet

if (-not $port8000) {
    Write-Host "  ✓ Port 8000 available" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Port 8000 in use" -ForegroundColor Yellow
}

if (-not $port5173) {
    Write-Host "  ✓ Port 5173 available" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Port 5173 in use" -ForegroundColor Yellow
}

# Check Critical Files
Write-Host "`n[6/6] Checking Critical Files..." -ForegroundColor Yellow
$files = @("LegalDOCAI\main.py", "LegalDoc-FrontEnd\src\App.jsx")
foreach ($f in $files) {
    if (Test-Path $f) {
        Write-Host "  ✓ $f" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $f missing" -ForegroundColor Red
        $issues++
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($issues -eq 0) {
    Write-Host "  ✅ All checks passed!" -ForegroundColor Green
    Write-Host "`nRun: .\start.ps1" -ForegroundColor Cyan
} else {
    Write-Host "  ❌ Found $issues issues" -ForegroundColor Red
    Write-Host "`nRun: .\setup.ps1" -ForegroundColor Cyan
}
Write-Host "========================================" -ForegroundColor Cyan
