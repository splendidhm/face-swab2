@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

title 얼굴 교체 프로그램

echo.
echo  🎭 얼굴 교체 프로그램
echo.
echo  ⚠️  사용 전 대상자의 명시적 동의 필요 — 교육/연구 목적일 때만 사용하십시오.
echo.
echo  GUI 프로그램을 시작합니다...
echo.

python face_swap_gui.py

if errorlevel 1 (
    echo.
    echo 오류가 발생했습니다. Python이 설치되어 있는지 확인해주세요.
    echo.
    pause
)
