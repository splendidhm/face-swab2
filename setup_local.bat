@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" goto install
if exist ".runtime\python\python.exe" (
    ".runtime\python\python.exe" -m venv .venv
) else (
    py -3.12 -m venv .venv
)
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements-local.lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_local.py
if errorlevel 1 goto failed
echo Setup complete. Run run_gui.bat.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.12 with Tk and check network access.
pause
exit /b 1
