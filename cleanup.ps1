# ============================================
# LegalDocAI - Cleanup Script
# ============================================
# Removes temporary files, caches, and resets development environment

param(
    [switch]$Deep,
    [switch]$Force
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LegalDocAI - Cleanup Utility" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $Force) {
    Write-Host "This will remove:" -ForegroundColor Yellow
    Write-Host "  - Python __pycache__ directories" -ForegroundColor Gray
    Write-Host "  - Temporary upload files" -ForegroundColor Gray
    Write-Host "  - Log files" -ForegroundColor Gray
    if ($Deep) {
        Write-Host "  - node_modules (--Deep)" -ForegroundColor Gray
        Write-Host "  - Python venv (--Deep)" -ForegroundColor Gray
        Write-Host "  - Build outputs (--Deep)" -ForegroundColor Gray
    }
    Write-Host ""
    $confirm = Read-Host "Continue? (y/N)"
    if ($confirm -ne "y") {
        Write-Host "Cleanup cancelled." -ForegroundColor Yellow
        exit 0
    }
}

$removed = 0

# ============ PYTHON CACHE ============
Write-Host "`n[1/6] Cleaning Python cache..." -ForegroundColor Yellow
$pycache = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $pycache) {
    Remove-Item -Recurse -Force $dir.FullName -ErrorAction SilentlyContinue
    $removed++
}
Write-Host "  ✓ Removed $($pycache.Count) __pycache__ directories" -ForegroundColor Green

# ============ PYTHON .pyc FILES ============
Write-Host "`n[2/6] Cleaning .pyc files..." -ForegroundColor Yellow
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue
foreach ($file in $pycFiles) {
    Remove-Item -Force $file.FullName -ErrorAction SilentlyContinue
    $removed++
}
Write-Host "  ✓ Removed $($pycFiles.Count) .pyc files" -ForegroundColor Green

# ============ TEMPORARY FILES ============
Write-Host "`n[3/6] Cleaning temporary files..." -ForegroundColor Yellow

# Remove temp uploads (keep processed)
if (Test-Path "LegalDOCAI\uploads") {
    $tempFiles = Get-ChildItem -Path "LegalDOCAI\uploads" -File -ErrorAction SilentlyContinue | 
                 Where-Object { $_.Name -match "^[0-9a-f\-]+" }
    foreach ($file in $tempFiles) {
        Remove-Item -Force $file.FullName -ErrorAction SilentlyContinue
        $removed++
    }
    Write-Host "  ✓ Removed $($tempFiles.Count) temporary upload files" -ForegroundColor Green
}

# Remove logs
$logFiles = Get-ChildItem -Path . -Recurse -Filter "*.log" -ErrorAction SilentlyContinue
foreach ($file in $logFiles) {
    Remove-Item -Force $file.FullName -ErrorAction SilentlyContinue
    $removed++
}
Write-Host "  ✓ Removed $($logFiles.Count) log files" -ForegroundColor Green

# ============ NODE CACHE ============
Write-Host "`n[4/6] Cleaning Node.js cache..." -ForegroundColor Yellow
if (Test-Path "LegalDoc-FrontEnd\.cache") {
    Remove-Item -Recurse -Force "LegalDoc-FrontEnd\.cache" -ErrorAction SilentlyContinue
    Write-Host "  ✓ Removed Vite cache" -ForegroundColor Green
}

# ============ DEEP CLEAN ============
if ($Deep) {
    Write-Host "`n[5/6] Deep clean (node_modules, venv, dist)..." -ForegroundColor Yellow
    
    # Remove node_modules
    if (Test-Path "LegalDoc-FrontEnd\node_modules") {
        Write-Host "  → Removing node_modules..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force "LegalDoc-FrontEnd\node_modules" -ErrorAction SilentlyContinue
        Write-Host "  ✓ Removed node_modules" -ForegroundColor Green
    }
    
    # Remove venv
    if (Test-Path "LegalDOCAI\venv") {
        Write-Host "  → Removing Python venv..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force "LegalDOCAI\venv" -ErrorAction SilentlyContinue
        Write-Host "  ✓ Removed venv" -ForegroundColor Green
    }
    
    # Remove dist
    if (Test-Path "LegalDoc-FrontEnd\dist") {
        Remove-Item -Recurse -Force "LegalDoc-FrontEnd\dist" -ErrorAction SilentlyContinue
        Write-Host "  ✓ Removed dist" -ForegroundColor Green
    }
} else {
    Write-Host "`n[5/6] Skipping deep clean (use -Deep flag)" -ForegroundColor Gray
}

# ============ PYTEST CACHE ============
Write-Host "`n[6/6] Cleaning test artifacts..." -ForegroundColor Yellow
$testDirs = @(".pytest_cache", "htmlcov", ".coverage")
foreach ($dir in $testDirs) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        Write-Host "  ✓ Removed $dir" -ForegroundColor Green
    }
}

# ============ SUMMARY ============
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Cleanup Complete! ✅" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($Deep) {
    Write-Host "Deep clean performed. Run .\setup.ps1 to reinstall dependencies." -ForegroundColor Yellow
} else {
    Write-Host "Standard cleanup complete." -ForegroundColor Green
    Write-Host "Use -Deep flag for full cleanup (removes node_modules, venv)" -ForegroundColor Gray
}
Write-Host ""
