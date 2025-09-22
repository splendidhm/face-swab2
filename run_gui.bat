@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================
echo    얼굴 교체 프로그램 GUI 실행
echo ========================================
echo.

REM 가상환경 확인 및 활성화
if exist "venv\Scripts\activate.bat" (
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
) else (
    echo 가상환경이 없습니다. 시스템 Python을 사용합니다.
)

echo.
echo GUI 프로그램을 시작합니다...
echo.

REM GUI 실행
python face_swap_gui.py

pause
