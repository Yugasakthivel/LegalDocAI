# ============================================
# LegalDocAI - Runtime Smoke Test
# ============================================
# Verifies that the project is operational, not just installed.

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LegalDocAI - Runtime Smoke Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

function Pass([string]$msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
    $script:passed++
}

function Fail([string]$msg) {
    Write-Host "  [ERR] $msg" -ForegroundColor Red
    $script:failed++
}

function Warn([string]$msg) {
    Write-Host "  [WARN] $msg" -ForegroundColor Yellow
}

# ============ TEST 1: Backend import sanity ============
Write-Host "[Test 1/8] Backend import sanity..." -ForegroundColor Yellow
Set-Location -Path "LegalDOCAI"

if (Test-Path "venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"

    $testScript = @'
import sys
sys.path.insert(0, '.')
import warnings

from backend.app.core.config import OPENAI_MODEL, MONGO_URI, DB_NAME
from backend.app.services.ml_service import classifier
import spacy
assert classifier is not None

try:
    spacy.load('en_core_web_sm')
except Exception as e:
    warnings.warn(f"spaCy model not loadable in this shell: {e}")

print('SUCCESS')
'@

    $tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmpPy -Value $testScript
    $result = python $tmpPy 2>&1
    Remove-Item -Path $tmpPy -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -eq 0 -and $result -match "SUCCESS") {
        Pass "Backend imports are operational"
    }
    else {
        Fail "Backend import test failed: $result"
    }
}
else {
    Fail "Backend virtual environment not found"
}

Set-Location -Path ".."
Write-Host ""

# ============ TEST 2: Frontend dependency sanity ============
Write-Host "[Test 2/8] Frontend dependency sanity..." -ForegroundColor Yellow
Set-Location -Path "LegalDoc-FrontEnd"

$criticalPaths = @(
    "node_modules\react\package.json",
    "node_modules\vite\package.json",
    "node_modules\react-router-dom\package.json"
)

$allExist = $true
foreach ($path in $criticalPaths) {
    if (-not (Test-Path $path)) {
        $allExist = $false
        break
    }
}

if ($allExist) {
    Pass "Frontend dependencies present"
}
else {
    Fail "Frontend dependencies missing (run npm install)"
}

Set-Location -Path ".."
Write-Host ""

# ============ TEST 3: Environment files ============
Write-Host "[Test 3/8] Environment files..." -ForegroundColor Yellow

$envChecks = @(
    "LegalDOCAI\.env",
    "LegalDoc-FrontEnd\.env"
)

$missingEnv = @()
foreach ($file in $envChecks) {
    if (-not (Test-Path $file)) {
        $missingEnv += $file
    }
}

if ($missingEnv.Count -eq 0) {
    Pass "Required .env files exist"
}
else {
    Fail "Missing env files: $($missingEnv -join ', ')"
}

Write-Host ""

# ============ TEST 4: Project scripts ============
Write-Host "[Test 4/8] Utility scripts..." -ForegroundColor Yellow

$scripts = @("setup.ps1", "start.ps1", "validate.ps1", "cleanup.ps1")
$missingScripts = @()
foreach ($script in $scripts) {
    if (-not (Test-Path $script)) {
        $missingScripts += $script
    }
}

if ($missingScripts.Count -eq 0) {
    Pass "Core utility scripts exist"
}
else {
    Fail "Missing scripts: $($missingScripts -join ', ')"
}

Write-Host ""

# ============ TEST 5: Backend health endpoint ============
Write-Host "[Test 5/8] Backend HTTP health..." -ForegroundColor Yellow

try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 10
    if ($resp.StatusCode -eq 200) {
        Pass "Backend health endpoint returned 200"
    }
    else {
        Fail "Backend health returned status $($resp.StatusCode)"
    }
}
catch {
    Fail "Backend health request failed: $($_.Exception.Message)"
    Warn "Start backend with .\\start.ps1 -BackendOnly"
}

Write-Host ""

# ============ TEST 6: Frontend Vite endpoint ============
Write-Host "[Test 6/8] Frontend HTTP check..." -ForegroundColor Yellow

try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5173/@vite/client" -UseBasicParsing -TimeoutSec 10
    if ($resp.StatusCode -eq 200) {
        Pass "Frontend Vite endpoint returned 200"
    }
    else {
        Fail "Frontend Vite endpoint returned status $($resp.StatusCode)"
    }
}
catch {
    Fail "Frontend request failed: $($_.Exception.Message)"
    Warn "Start frontend with .\\start.ps1 -FrontendOnly"
}

Write-Host ""

# ============ TEST 7: Frontend process identity ============
Write-Host "[Test 7/8] Frontend process identity..." -ForegroundColor Yellow

$portLine = netstat -ano | Select-String "^\s*TCP\s+.*:5173\s+.*LISTENING\s+\d+\s*$" | Select-Object -First 1
if ($portLine) {
    $tokens = ($portLine -replace "^\s+", "") -split "\s+"
    $pidNum = $tokens[-1]

    if ($pidNum -match "^\d+$") {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pidNum"
            if ($null -ne $proc -and $proc.CommandLine -match "vite") {
                Pass "Port 5173 is served by LegalDoc-FrontEnd Vite process"
            }
            else {
                Warn "Port 5173 is listening, but command line is not clearly a Vite process"
                Fail "Frontend process identity is uncertain"
            }
        }
        catch {
            Fail "Could not inspect process for PID $pidNum"
        }
    }
    else {
        Fail "Could not parse PID from netstat output"
    }
}
else {
    Fail "No process found listening on 5173"
}

Write-Host ""

# ============ TEST 8: Core analysis endpoint ============
Write-Host "[Test 8/8] Core analysis endpoint (/api/analysis/advanced)..." -ForegroundColor Yellow

Set-Location -Path "LegalDOCAI"
if (Test-Path "venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

$analysisScript = @'
import requests

url = 'http://localhost:8000/api/analysis/advanced'
data = {'text': 'This is a short legal test text for smoke testing.'}

resp = requests.post(url, data=data, timeout=60)
print(resp.status_code)
print(resp.text[:200])
'@

try {
    $tmpPyAnalysis = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmpPyAnalysis -Value $analysisScript
    $analysisOut = python $tmpPyAnalysis 2>&1
    Remove-Item -Path $tmpPyAnalysis -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -eq 0 -and $analysisOut -match "^200") {
        Pass "Analysis endpoint returned 200"
    }
    else {
        Fail "Analysis endpoint failed: $analysisOut"
    }
}
catch {
    Fail "Analysis endpoint test failed: $($_.Exception.Message)"
}

Set-Location -Path ".."
Write-Host ""

# ============ SUMMARY ============
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Smoke Test Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Passed: $passed / 8" -ForegroundColor Green
Write-Host "  Failed: $failed / 8" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($failed -eq 0) {
    Write-Host "Project runtime smoke test passed." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "Project runtime smoke test failed. Fix the failed checks above." -ForegroundColor Yellow
    exit 1
}
