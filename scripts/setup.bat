@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM Face Swap 환경 설정 스크립트

echo Face Swap 환경 설정
echo ===================

REM Python 버전 확인
python --version
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다.
    echo Python 3.8 이상을 설치해주세요.
    pause
    exit /b 1
)

REM 가상환경 생성
echo 가상환경 생성 중...
python -m venv venv
if errorlevel 1 (
    echo 가상환경 생성 실패
    pause
    exit /b 1
)

REM 가상환경 활성화
echo 가상환경 활성화 중...
call venv\Scripts\activate.bat

REM pip 업그레이드
echo pip 업그레이드 중...
python -m pip install --upgrade pip

REM 의존성 설치
echo 의존성 설치 중...
pip install -r requirements.txt
if errorlevel 1 (
    echo 의존성 설치 실패
    pause
    exit /b 1
)

REM 디렉토리 생성
echo 디렉토리 생성 중...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir checkpoints 2>nul
mkdir logs 2>nul

REM 설정 파일 복사
echo 설정 파일 설정 중...
if not exist "configs\default.yaml" (
    copy "configs\deepfacelab.yaml" "configs\default.yaml" 2>nul
)

echo 환경 설정 완료!
echo ================
echo 사용법:
echo 1. scripts\run.bat 실행
echo 2. 또는 개별 스크립트 실행
echo.
echo 예시:
echo python scripts\preprocess.py --source "source.mp4" --target "target.mp4" --output "data\processed"
echo python scripts\train.py --model "fomm" --data "data\processed" --output "checkpoints"
echo python scripts\inference.py --model "fomm" --source "source.jpg" --target "target.mp4" --output "result.mp4" --checkpoint "checkpoints\best.pth"
echo.
pause
