# run.ps1
# ----------------------------------------
# Activate virtual environment
# ----------------------------------------
Write-Host "Activating virtual environment..."
& D:/Project/LegalDocAI/venv/Scripts/Activate.ps1

# ----------------------------------------
# Set current folder as working directory
# ----------------------------------------
Set-Location "D:\Project\LegalDocAI\LegalDOCAI"

# ----------------------------------------
# Add current folder to PYTHONPATH
# ----------------------------------------
$env:PYTHONPATH = Get-Location
Write-Host "PYTHONPATH set to $env:PYTHONPATH"

# ----------------------------------------
# Start FastAPI app with uvicorn
# ----------------------------------------
Write-Host "🚀 Starting FastAPI app..."
uvicorn main:app --reload --host 127.0.0.1 --port 8000
