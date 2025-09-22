@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

title 얼굴 교체 프로그램

echo.
echo  ███████╗ █████╗  ██████╗███████╗    ███████╗██╗    ██╗ █████╗ ██████╗ 
echo  ██╔════╝██╔══██╗██╔════╝██╔════╝    ███████║██║    ██║██╔══██╗██╔══██╗
echo  █████╗  ███████║██║     █████╗      ██╔════╝██║ █╗ ██║███████║██████╔╝
echo  ██╔══╝  ██╔══██║██║     ██╔══╝      ███████╗██║███╗██║██╔══██║██╔═══╝ 
echo  ██║     ██║  ██║╚██████╗███████╗    ╚════██║╚███╔███╔╝██║  ██║██║     
echo  ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝    ███████║ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝     
echo.
echo  ⚠️  사용 전 대상자의 명시적 동의 필요 — 교육/연구 목적일 때만 사용하십시오.
echo.
echo  실행 방법을 선택하세요:
echo  1. GUI 모드 (권장)
echo  2. 명령줄 모드
echo  3. 종료
echo.

set /p choice="선택 (1-3): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto end

echo 잘못된 선택입니다.
goto end

:gui
echo.
echo GUI 모드를 시작합니다...
python face_swap_gui.py
goto end

:cli
echo.
echo 명령줄 모드를 시작합니다...
scripts\run.bat
goto end

:end
echo.
echo 프로그램을 종료합니다.
pause
