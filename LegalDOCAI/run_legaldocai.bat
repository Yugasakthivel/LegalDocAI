@echo off
echo Activating LegalDocAI environment...

REM go to project root
cd /d D:\Project\LegalDocAI

REM activate venv
call venv\Scripts\activate.bat

REM go to app folder
cd LegalDOCAI

echo Starting LegalDocAI API...
python main.py

pause
