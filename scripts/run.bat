@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM Face Swap 실행 스크립트

echo Face Swap Pipeline 실행 스크립트
echo ================================

REM Python 경로 설정
set PYTHONPATH=%CD%

REM 가상환경 활성화 (있는 경우)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo 가상환경 활성화됨
)

REM 실행 모드 선택
echo 실행할 모드를 선택하세요:
echo 1. 얼굴 추출 (Face Extraction)
echo 2. 얼굴 교체 (Face Swap)
echo 3. 학습 (Train)
echo 4. 추론 (Inference)
echo 5. 후처리 (Postprocess)
echo 6. 오디오 처리 (Audio)
echo 7. 전체 파이프라인 (Full Pipeline)
echo 8. 종료

set /p choice="선택 (1-8): "

if "%choice%"=="1" goto extract_faces
if "%choice%"=="2" goto swap_faces
if "%choice%"=="3" goto train
if "%choice%"=="4" goto inference
if "%choice%"=="5" goto postprocess
if "%choice%"=="6" goto audio
if "%choice%"=="7" goto full
if "%choice%"=="8" goto end

echo 잘못된 선택입니다.
goto end

:extract_faces
echo 얼굴 추출 모드
set /p video="비디오 파일 경로: "
set /p output="출력 디렉토리: "
set /p sample_rate="샘플링 비율 (기본값: 3): "
if "%sample_rate%"=="" set sample_rate=3
python scripts\extract_faces.py --video "%video%" --sample-rate %sample_rate% --out "%output%"
goto end

:swap_faces
echo 얼굴 교체 모드
set /p mode="모드 선택 (lightwarp/fomm): "
set /p target="타겟 이미지 경로: "
set /p video="소스 비디오 경로: "
set /p meta="메타데이터 JSON 경로: "
set /p output="출력 비디오 경로: "
if "%mode%"=="fomm" (
    set /p checkpoint="체크포인트 파일 경로: "
    python scripts\swap_faces.py --mode %mode% --target "%target%" --video "%video%" --meta "%meta%" --out "%output%" --checkpoint "%checkpoint%"
) else (
    python scripts\swap_faces.py --mode %mode% --target "%target%" --video "%video%" --meta "%meta%" --out "%output%"
)
goto end

:train
echo 학습 모드
set /p model="모델 타입 (deepfacelab/fomm): "
set /p data="학습 데이터 경로: "
set /p output="체크포인트 출력 디렉토리: "
python scripts\train.py --model "%model%" --data "%data%" --output "%output%"
goto end

:inference
echo 추론 모드
set /p model="모델 타입 (deepfacelab/fomm): "
set /p source="소스 비디오/이미지 경로: "
set /p target="타겟 비디오 경로: "
set /p output="출력 비디오 경로: "
set /p checkpoint="체크포인트 파일 경로: "
python scripts\inference.py --model "%model%" --source "%source%" --target "%target%" --output "%output%" --checkpoint "%checkpoint%"
goto end

:postprocess
echo 후처리 모드
set /p input="입력 비디오 경로: "
set /p output="출력 비디오 경로: "
set /p target="타겟 비디오 경로 (선택적): "
if "%target%"=="" (
    python scripts\postprocess.py --input "%input%" --output "%output%"
) else (
    python scripts\postprocess.py --input "%input%" --output "%output%" --target "%target%"
)
goto end

:audio
echo 오디오 처리 모드
set /p input="입력 비디오 경로: "
set /p output="출력 비디오 경로: "
set /p enhancement="향상 적용? (y/n): "
if "%enhancement%"=="y" (
    python scripts\audio.py --input "%input%" --output "%output%" --enhancement
) else (
    python scripts\audio.py --input "%input%" --output "%output%"
)
goto end

:full
echo 전체 파이프라인 모드
set /p source="소스 비디오 경로: "
set /p target="타겟 비디오 경로: "
set /p output="출력 비디오 경로: "
set /p model="모델 타입 (deepfacelab/fomm): "
python scripts\run_full_pipeline.py --source "%source%" --target "%target%" --output "%output%" --model "%model%"
goto end

:end
echo 작업 완료!
pause
