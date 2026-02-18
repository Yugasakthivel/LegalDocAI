# Deployment Status

Snapshot date: 2026-02-15
Type: Local machine status snapshot (not continuous monitoring)

## Verified Checks

- `validate.ps1` executed successfully (exit code `0`).
- Backend health endpoint responded with HTTP `200`:
  - `GET http://localhost:8000/api/health`
- Frontend root responded with HTTP `200`:
  - `GET http://localhost:5173/`
- Port usage detected:
  - `8000` listening (Python process)
  - `5173` listening (Node process)
- Frontend process identity verified:
  - Process: `node.exe`
  - Command: `D:\Legal-mohan\Legaldoc-new\LegalDoc-FrontEnd\node_modules\vite\bin\vite.js`

## Current Interpretation

- Backend: reachable and healthy.
- Frontend: reachable and confirmed to be this repository's Vite app.

## How to Re-Verify Quickly

```powershell
cd d:\Legal-mohan\Legaldoc-new
.\validate.ps1
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
curl.exe -I http://localhost:5173/
netstat -ano | findstr :5173
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object ProcessId,CommandLine
```
