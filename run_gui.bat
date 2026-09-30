@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_local.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" face_swap_gui.py
if errorlevel 1 pause
