# FaceSwab2 작업 인수인계

최종 기록일: 2026-10-01 (Asia/Seoul)

이 문서는 다른 작업환경의 개발자/에이전트가 현재 구현을 이어가기 위한 기록이다. 작업을 시작할 때 이 문서와 `README.md`, `LOCAL_GUIDE.ko.md`를 먼저 읽는다. 파일명은 사용자 요청에 따라 **AGENT.md**이다. 도구가 `AGENTS.md`만 자동 탐색하는 경우에는 이 파일을 명시적으로 열어야 한다.

## 1. 저장소 및 현재 작업 상태

- 원격 저장소: https://github.com/splendidhm/face-swab2.git
- 프로젝트 폴더: `C:\Users\케이지에프앤비\codex\projects\faceswab2`
- 현재 작업 브랜치: `feature/natural-face-expression`. 안정 브랜치: `main`.
- 작업 시작 기준 HEAD: `33ed48659a4a04d1f7c159ddb99a9610f1886f3f`
- 기준 커밋: 2025-09-22, `Initial commint : first`
- 안정 기준 버전: **`v0.1.0`**, `main`에 반영하는 첫 로컬 얼굴 합성 리팩토링.
- 사용자는 현재 작업을 안정 기준으로 보존하고 이후 업데이트/버그 수정은 별도 브랜치에서 관리하도록 지시했다.
- `v0.1.0` 안정 기준은 `3240b286fbf346fa07a105bf36d30d5b97ad173c`이며 원격 반영 완료. 업데이트 1은 별도 기능 브랜치의 작업이며 상세는 14절을 따른다.

### 다른 환경으로 전달할 때

원격 저장소를 clone한 뒤 `v0.1.0` 태그로 안정 기준을 재현할 수 있다. 추가 로컬 변경이 있다면 수정 파일과 새 파일을 모두 확인한다. `git diff`만 저장하면 추적되지 않은 새 파일은 빠진다.

```powershell
git status --short
git ls-files --others --exclude-standard
git fetch origin --tags
git show v0.1.0 --stat
```

`.runtime/`, `.venv/`, `models/`, `workspace/`, `outputs/`는 새 `.gitignore`에 포함되어 있다. 가상환경을 다른 경로로 그대로 복사하지 말고 대상 환경에서 재생성한다. 모델은 설치 스크립트로 다시 받을 수 있다. 사용자 입력·출력 및 테스트 이미지는 필요할 때 별도로 전달한다. 기존 저장소가 이미 추적하던 pycache/임시 산출물은 `.gitignore` 추가만으로 추적 해제되지 않았으며 아직 정리하지 않았다.

## 2. 사용자 요구사항 및 구현 방향

요구사항:

1. 기존 영상의 캐릭터 얼굴을 준비한 얼굴 이미지로 변환.
2. JPEG/PNG 업로드 시 정면 얼굴만 자동 인식하여 누끼 추출.
3. 영상에서 드래그한 얼굴에만 합성하고 움직임을 추적.
4. 720p MP4 압축 영상 및 스테레오 사운드.
5. 로컬 PC에서 실행 가능해야 함.

현재 구현은 **얼굴 윤곽 누끼 + 랜드마크 기반 기하 변형 + 선택 영역 추적** 방식이다. 사전 조사에서는 FaceFusion/ONNX 엔진 연동도 후보로 제안했으나, 실제 구현에는 FaceFusion/INSwapper/HyperSwap을 통합하지 않았다. 생성형 신원 교체 또는 딥러닝 기반 립싱크가 구현되었다고 설명하면 안 된다.

## 3. 완료한 작업

- 저장소 복제 및 기존 코드 점검: `PROJECT_AUDIT.ko.md`.
- 합성 대신 2초 대기하던 GUI를 실제 로컬 합성 엔진으로 교체.
- JPEG/PNG 형식 확인, EXIF 회전 적용, 정면 얼굴 한 명 검출.
- 얼굴 외곽 랜드마크로 알파 마스크 생성, 부드러운 경계 처리, 기존 PNG 투명도 반영.
- 투명 얼굴 미리보기 및 PNG 저장.
- 영상 가져오기 시 비율을 유지한 1280×720, 30fps MP4 변환. 세로 영상은 여백 추가.
- 프레임 슬라이더 및 이전/다음 프레임 탐색.
- 드래그 선택 및 표시 좌표→실제 영상 좌표 변환. 역방향 드래그 지원.
- 선택 시점의 합성 미리보기.
- 여러 시점의 얼굴 선택을 저장하고 각 시점에서 추적을 재시작.
- 사람형 얼굴: 지역 얼굴 재검출, CSRT, optical flow, 랜드마크 변형.
- 자동 검출이 어려운 캐릭터: 선택 영역 추적 모드로 전환. 강제 영역 모드도 제공.
- 추적 상실/장면 전환 시 합성 중단, 원본 유지, 실패 구간 기록.
- 작업 진행률, 취소, GUI worker→메인 스레드 큐 연결.
- FFmpeg로 H.264 MP4 / AAC 스테레오 출력 및 파일 검증.
- 동일 엔진을 사용하는 CLI 추가.
- Python/가상환경/모델/FFmpeg 설치, 고정 의존성 파일 및 설치 스크립트 작성.
- 영상·모델 테스트 12개, GUI 테스트 3개 통과.

## 4. 코드 지도

| 파일/폴더 | 역할 |
|---|---|
| `face_swap_gui.py` | 새 GUI 진입점. `FaceSwapGUI`, `main`을 가져옴 |
| `face-swab2.py` | 비어 있던 파일을 새 GUI 진입점으로 연결 |
| `src/local/gui.py` | 입력 선택, 영상 탐색, 드래그, 미리보기, worker 큐, 저장 UI |
| `src/local/faces.py` | Tasks 모델 래퍼, 정면 판정, 누끼 추출, 고정 메시 변형과 합성 연결 |
| `src/local/matting.py` | 업데이트 1: 얼굴/헤어 분할, 곡선 윤곽, guided filter, 안쪽 페더링 |
| `src/local/expression.py` | 업데이트 1: 468점/852삼각형 고정 메시, 표정 필터, 눈/입안 보존, LAB/다중 해상도 합성 |
| `src/local/tracking.py` | 선택 ROI의 CSRT 추적, 지역 얼굴 재검출, optical flow, 추적 상실 판단 |
| `src/local/media.py` | 경로 기준, 영상 정보/프레임 읽기, FFmpeg 실행/취소, 720p 변환, 오디오 결합 |
| `src/local/pipeline.py` | 선택 시점별 추적 및 프레임 합성, 임시파일 처리, 결과 검증과 보고서 |
| `scripts/local_swap.py` | 새 엔진의 CLI |
| `scripts/setup_local.py` | 공식 모델 다운로드, SHA256 확인, Python/Tk/FFmpeg/모델 환경 진단 |
| `setup_local.bat` | 가상환경 준비 및 lock 설치, 모델 준비 |
| `run_gui.bat` | 작업 디렉터리 고정, UTF-8 설정, `.venv`로 GUI 실행 |
| `start.bat`, `실행.bat`, `얼굴교체.bat` | 모두 `run_gui.bat`로 연결 |
| `scripts/setup.bat` | 새 `setup_local.bat`로 연결 |
| `requirements.txt` | 새 로컬 lock 파일을 포함하는 기본 설치 파일 |
| `requirements-local.txt` | 직접 의존성 목록 |
| `requirements-local.lock.txt` | 설치·검증한 전체 Python 패키지 버전 |
| `requirements-legacy.txt` | 이전 torch/TensorFlow/dlib 등의 학습 의존성 보관 |
| `tests/test_local_pipeline.py` | 실제 모델 및 영상 입출력 회귀 테스트 12개 |
| `tests/test_local_gui.py` | 숨긴 Tk 창을 이용한 GUI 통합 테스트 3개 |
| `README-legacy.md` | 이전 README 보관. 새 실행 안내가 아님 |

`src/infer/`, `src/fomm/`, `src/deepfacelab/`, `src/cli.py`와 예전 학습/합성 스크립트는 보관되어 있지만 새 GUI는 이 경로를 사용하지 않는다. 예전 `scripts/swap_faces.py`와 새 `scripts/local_swap.py`를 혼동하지 않는다. 구형 엔진 오류는 새 엔진으로 우회했으며 모두 수정한 것이 아니다.

## 5. 데이터 흐름과 유지해야 할 계약

```text
원본 영상 → normalize_video → 1280×720 / 30fps 준비 영상
JPEG/PNG → extract_face → FaceAsset + 투명 PNG
준비 영상의 시점 선택 → 드래그 ROI → selections
render_video → 선택 지점에서 SelectedTracker 초기화
            → 매 프레임 추적 / 얼굴 변형 / 알파 합성
            → 임시 영상 → 원본 오디오 결합 및 H.264 인코딩
            → 출력 검증 → MP4 + report.json
```

주요 인터페이스:

- `FaceAsset`: BGR 이미지, uint8 알파, 랜드마크 좌표, 고정 얼굴 메시 삼각형 인덱스, PNG 경로.
- `selections`: Python에서는 `{프레임번호(int): [x, y, width, height]}`. JSON 저장 시 키는 문자열이 되므로 CLI에서 정수로 변환한다.
- 좌표는 **원본 파일 해상도가 아니라 정규화한 1280×720 영상 기준**이다.
- 프레임 번호는 0부터 시작하고 정규화한 30fps 기준이다.
- GUI 표시 크기는 832×468이다. 좌표 변환 배율은 `1280 / 832`.
- 첫 선택 시점 전에는 합성하지 않는다. 이전 구간을 역방향 추적하는 기능은 없다.
- 여러 선택 시점은 각 시점에서 추적을 다시 시작하는 명시적 지점이다. 여러 얼굴을 동시에 교체하는 기능은 아니다.
- 짧은 랜드마크 검출 실패 때는 해당 프레임의 합성을 건너뛰고 같은 ROI에서 재검출한다. 5프레임을 넘는 지속 실패는 다음 선택 지점까지 추적 상실 상태로 남는다.
- `composite()`는 원본 프레임을 직접 수정하지 않고 결과를 반환한다. 이미지/마스크/랜드마크의 좌표계를 일치시킨다.
- Tk 위젯 접근은 메인 스레드에서 수행한다. 긴 작업 결과는 `queue.Queue`로 전달한다.
- 취소는 `threading.Event`를 사용하며 FFmpeg 하위 프로세스도 정리한다.
- CLI와 GUI는 `render_video()`를 공유해야 한다.

출력 계약:

| 항목 | 현재 값 |
|---|---|
| 컨테이너 | MP4 |
| 영상 | 1280×720, 30fps, H.264, yuv420p, CRF 20, faststart |
| 비율 | 종횡비 유지 + letterbox, SAR 1:1 |
| 오디오 | AAC, 48kHz, 192kbps, 스테레오 2채널 |
| 무음 입력 | 무음 스테레오 트랙 생성 |
| 모노 입력 | 스테레오로 변환 |
| 짧은 오디오 | 무음으로 채우고 영상 길이에 맞춤 |
| 보고서 | `<출력 이름>.report.json` |

보고서 필드: `output`, `frames`, `replaced_frames`, `unchanged_before_selection`, `skipped_tracking_ranges`, `modes`, `selections`, `fps`, `resolution`, `video_codec`, `audio_codec`, `audio_channels`, `audio_sample_rate`.

출력은 같은 파일시스템의 작업별 임시 폴더에서 만든 뒤 검증하고 `os.replace()`한다. 영상과 보고서는 각각 교체하므로 두 파일 전체가 하나의 원자적 트랜잭션인 것은 아니다.

## 6. 사용 라이브러리와 런타임

직접 의존성:

| 라이브러리 | 검증 버전 | 사용 목적 |
|---|---|---|
| Python | 3.12.10, Windows x64 | 실행 환경 |
| Tcl/Tk / tkinter | 8.6 / Python 표준 라이브러리 | 데스크톱 GUI |
| MediaPipe | 0.10.32 | Tasks FaceLandmarker, 얼굴 랜드마크 |
| opencv-contrib-python | 4.11.0.86 | CSRT, optical flow, 프레임 입출력, 변형·합성 |
| NumPy | 2.2.6 | 이미지·마스크·좌표 배열 연산 |
| Pillow | 11.3.0 | JPEG/PNG 읽기, EXIF 회전, RGBA 저장, Tk 이미지 |
| SciPy | 1.15.3 | 곡선 윤곽 보간(CubicSpline). 이전 버전은 Delaunay 사용 |
| imageio-ffmpeg | 0.6.0 | 배포에 포함된 FFmpeg 실행파일 경로 |
| FFmpeg | Windows 번들 7.1 | 영상 정규화, H.264/AAC 인코딩, 오디오 처리 |

표준 라이브러리: `threading`, `queue`, `subprocess`, `tempfile`, `pathlib`, `json`, `dataclasses`, `argparse`, `hashlib`, `urllib.request`, `unittest` 등.

전체 전이 의존성 버전은 `requirements-local.lock.txt`를 기준으로 한다. 설치된 항목에는 absl-py, cffi, contourpy, cycler, flatbuffers, fonttools, kiwisolver, matplotlib, packaging, pycparser, pyparsing, python-dateutil, six, sounddevice도 있다. matplotlib/sounddevice는 새 앱의 핵심 합성 코드에서 직접 사용하지 않는다.

주의 사항:

- `mp.solutions` 구형 API를 사용하지 않는다. 새 코드는 `mp.tasks.vision.FaceLandmarker` 기반이다.
- CSRT를 위해 **contrib** OpenCV가 필요하다. `opencv-python`과 `opencv-contrib-python`을 함께 설치하지 않는다.
- 새 GUI에 torch/TensorFlow/dlib/FaceFusion/ONNX Runtime 의존성은 없다.
- 별도 시스템 FFmpeg/ffprobe 설치 없이 `imageio_ffmpeg.get_ffmpeg_exe()`를 사용한다.
- 버전 변경 시 실제 모델 및 영상 회귀 테스트를 다시 수행하고 lock 파일을 함께 갱신한다.

## 7. 모델 정보

- 파일: `models/face_landmarker.task`
- 모델: Google MediaPipe Face Landmarker float16 v1
- URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- 다운로드 및 검증: `scripts/setup_local.py`
- API 문서: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

Tasks 모델 로딩에는 경로 대신 `model_asset_buffer=MODEL.read_bytes()`를 사용한다. 한글 경로 환경에서 테스트했다. 설치 이후 영상/사진은 외부로 업로드하지 않고 로컬 CPU에서 처리한다.

## 8. 다른 환경에서 재개하는 절차

### Windows 권장 경로

1. 저장소를 clone하고 `main` 또는 `v0.1.0` 태그를 확인한다. 안정 버전에서 개발을 시작하려면 `git switch -c feature/<작업명> v0.1.0`처럼 새 브랜치를 만든다.
2. Python 3.12 x64를 설치한다. pip, venv, Tcl/Tk 및 필요하면 Windows Python Launcher를 포함한다.
3. 프로젝트 루트에서 `setup_local.bat`를 실행한다.
4. `run_gui.bat`를 실행한다.

`setup_local.bat`는 기존 `.venv`가 없으면 `.runtime/python/python.exe`를 우선 사용하고, 없으면 `py -3.12`를 사용한다. 두 경로를 모두 사용할 수 없다면 직접 Python 3.12 실행파일로 가상환경을 만든다.

```powershell
# 프로젝트 루트에서 실행. Python Launcher 사용 예시.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.lock.txt
.\.venv\Scripts\python.exe scripts\setup_local.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe face_swap_gui.py
```

처음 패키지/모델을 내려받을 때 네트워크가 필요하다. 원래 PC의 `.runtime/python`은 서명을 확인한 python.org 설치 프로그램으로 준비했지만, 해당 설치 프로그램/런타임은 Git에 포함되지 않는다.

macOS/Linux에서는 `.bat` 파일을 그대로 사용할 수 없다. 해당 OS의 Python/Tk 설치와 실행 명령을 별도로 구성해야 한다. 다른 OS에서의 모델·코덱·GUI 동작은 아직 검증하지 않았다.

### CLI 사용 예시

```powershell
.\.venv\Scripts\python.exe scripts\local_swap.py --video "input.mp4" --face "face.png" --roi 400 180 180 220 --start-frame 0 --out "outputs\result.mp4"
```

- `--region-mode`: 얼굴 검출 여부와 관계없이 사각형 영역 모드.
- `--selections result.report.json`: 보고서의 선택 지점 재사용. `--roi` 대신 지정한다.
- GUI에서 선택한 좌표를 원본 해상도에 재해석하지 않는다.

## 9. 테스트와 검증 기록

2026-09-30 기존 환경에서 통과한 결과이다. 안정 기준 반영 시 전체 `test_local_*.py`를 다시 실행하여 15개 테스트 통과와 의존성 충돌 없음을 확인한다.

- `test_local_pipeline.py`: 12개 통과.
- `test_local_gui.py`: 3개 통과.
- `python -m compileall`: 새 모듈과 진입점 구문 검사 통과.
- `python -m pip check`: 의존성 충돌 없음.
- `git diff --check`: 공백 오류 없음. Windows 줄바꿈 변환 안내는 표시될 수 있음.
- 실제 출력 MP4의 H.264/AAC/48kHz/stereo 메타데이터 및 FFmpeg 전체 디코딩 확인.

회귀 테스트 항목:

1. 정면 얼굴 추출 및 PNG 알파.
2. JPEG 지원 및 얼굴 없는 이미지 거부.
3. 선택 얼굴만 변경하고 다른 얼굴 영역 보존.
4. 움직이는 얼굴 추적.
5. 장면 전환 후 임의의 다른 얼굴로 재연결하지 않음.
6. 강제 영역 모드.
7. 선택 이전 구간 유지, 전체 길이 및 720p/H.264/무음 스테레오 출력.
8. 모노 유음 오디오의 스테레오 변환.
9. 취소 및 입력 영상 덮어쓰기 차단.
10. 세로 영상 letterbox.
11. 장면 전환 후 추가 선택 지점에서 합성 재개.
12. 비인간형 그림의 자동 영역 모드 및 이동 추적.
13. GUI 역방향 드래그 좌표 변환 및 선택 지점 관리.
14. 너무 작은 드래그 차단.
15. worker 완료 결과를 메인 큐에서 처리.

### 테스트 데이터 복원 및 실행

```powershell
New-Item -ItemType Directory -Force workspace/test-assets | Out-Null
curl.exe -fL "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png" -o "workspace/test-assets/astronaut.png"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_local_*.py" -v
```

얼굴 모델 또는 astronaut 이미지가 없으면 모델/영상 테스트 클래스가 **skip**된다. 테스트 실행이 성공했다는 메시지만 보지 말고 15개가 실제 실행되었는지 확인한다. GUI 테스트는 숨긴 Tk 창을 만들므로 GUI 세션/Tk를 사용할 수 있는 환경이 필요하다.

기존 `tests/test_simple.py`, `tests/test_lightwarp.py`, `tests/test_color_transfer.py`는 이전 엔진용이다. 새 최소 환경에서 전체 `tests/`를 무조건 실행하여 얻는 결과를 새 엔진의 검증 결과와 혼동하지 않는다.

## 10. 제약과 미검증 사항

- 실제 사용자가 준비한 얼굴 이미지/캐릭터 영상으로 품질 평가하지 않았다. 공개 astronaut 이미지 기반 짧은 이동 영상과 간단한 그림을 사용했다.
- GUI는 코드 수준 통합 테스트를 수행했다. 수동 시각/사용성 검증, 다양한 Windows 배율/화면 크기 검증은 미완료다.
- 정면 판정은 눈/코 상대 위치의 휴리스틱이다. 정확한 3D pose 추정이 아니다.
- 업데이트 1은 얼굴/헤어 분할을 추가했다. 다만 머리카락에 가려진 피부를 생성하거나 모든 가림을 복구하지는 않는다.
- 영역 모드는 얼굴을 선택 사각형에 맞춰 리사이즈한다. 고개 회전/표정 전이 기능이 없다.
- identity embedding 기반 추적이 없어서 유사 얼굴의 교차·긴 가림에서 인물 신원 유지가 보장되지 않는다.
- 손/물체 가림 분할, 생성형 얼굴 복원, 립싱크, 전신 추적은 없다.
- 첫 선택 이전의 역방향 추적, 여러 얼굴 동시 교체는 없다.
- 업데이트 1의 중간 영상은 FFV1 무손실이다. 입력 정규화 및 최종 H.264 단계의 손실 압축은 유지된다.
- 긴 영상의 성능/최대 메모리/디스크 사용량, 고해상도 원본의 대규모 처리 벤치마크는 미실시다.
- Intel GPU 가속은 구현하지 않았으며 CPU로 동작한다.
- `workspace/`에 준비 영상/얼굴 캐시가 남는다. 자동 용량 관리 UI는 없다.
- 원래 저장소의 README는 교육/연구 목적 사용을 명시하며 별도 LICENSE 파일은 발견하지 못했다. 배포 코드와 향후 추가 모델의 사용 조건을 별도로 확인해야 한다.

## 11. 다음 작업의 권장 순서

1. 새 환경에서 두 모델 설치와 25개 회귀 테스트를 재현한다.
2. 실제 사용자의 짧은 샘플 영상과 얼굴로 결과를 확인하고, 추적 실패 프레임/왜곡/가림을 구체적으로 수집한다.
3. GUI의 실제 드래그·프레임 이동·미리보기·취소·저장 흐름을 수동 확인한다.
4. 요구 품질에 따라 회전 추정/가림 마스크/신원 추적 또는 생성형 교체 엔진을 검토한다. 현재 누끼 방식과 별개인 작업임을 명확히 한다.
5. FFV1 임시 파일의 디스크 사용량, 긴 영상 성능, 캐시 관리 개선을 검토한다.
6. 필요 시 Intel 가속을 도입하고 동일 입력에서 CPU와 품질·속도를 비교한다.
7. 아래 브랜치 규칙에 따라 변경을 전달한다. 기존 추적 산출물 정리는 별도 변경으로 검토한다.

작업 중에는 로컬 입력 보존, 선택한 얼굴만 처리, 추적 실패 시 원본 유지, 실제 파일 검증 후 완료 표시, GUI 메인 스레드 규칙을 유지한다. 합성 품질을 개선하더라도 실패를 성공처럼 표시하거나 원본 프레임을 조용히 누락시키지 않는다.

## 12. 원래 개발 PC 참고

- Windows / PowerShell, 사용자 경로에 한글 포함.
- CPU: Intel Core i5-1240P.
- RAM: 약 16GB.
- GPU: Intel Iris Xe Graphics, 조회 드라이버 30.0.101.1298.
- NVIDIA GPU는 조회되지 않았으며 CUDA 환경을 구성하지 않았다.
- Git: 2.51.0.windows.1.
- Python 설치 경로: 프로젝트 `.runtime/python`.
- 실행 가상환경: 프로젝트 `.venv`.

위 정보는 기존 PC의 검증 조건이다. 새 환경에서는 프로젝트 절대경로를 강제로 동일하게 맞출 필요가 없으며, 코드의 `Path(__file__)` 기준 경로와 런처의 작업 디렉터리 고정을 유지하면 된다.

## 13. 안정 버전 및 이후 브랜치 운영 규칙

이 규칙은 현재 버전을 안정 기준으로 두고 후속 작업은 별도 브랜치에서 관리하겠다는 사용자의 지시를 반영한다.

- `main`: 검증된 안정 코드. 후속 개발을 시작하기 전에 작업 브랜치로 전환한다.
- `v0.1.0`: 현재 안정 기준을 보존하는 annotated tag. 이후 강제 이동/덮어쓰기를 하지 않는다.
- `feature/<주제>`: 기능 추가와 업데이트.
- `fix/<주제>`: 버그 수정. 긴급 수정도 `main`에 직접 개발하지 않는다.
- 필요 시 `docs/<주제>` 또는 `chore/<주제>`: 문서와 유지보수 변경.
- 변경별로 작은 커밋을 만들고 `CHANGELOG.md`의 Unreleased 항목 및 필요하면 인수인계 문서를 갱신한다.
- 해당 변경의 테스트를 수행하고, 원격 작업 브랜치에 push한 뒤 PR/리뷰를 거쳐 `main`에 병합하는 방식을 기본으로 한다. 이후 병합·푸시는 해당 작업에서 부여된 사용자 권한 범위를 따른다.
- 이후 안정 릴리스는 별도 버전 태그로 남긴다. 회귀 테스트 통과만으로 실제 영상 품질까지 보장된 것으로 기록하지 않는다.
- 이 문서를 자동으로 찾기 위한 `AGENTS.md`가 루트에 추가되어 있다.

```powershell
git fetch origin --tags
git switch main
git pull --ff-only origin main
git switch -c feature/<작업명>
# 버그 수정은 fix/<작업명>
```

작업 시작 전 미커밋 변경이 있다면 덮어쓰거나 임의로 reset하지 말고 해당 변경을 보존한다.

## 14. 업데이트 1 — 누끼 경계와 정밀 표정 (2026-10-01)

작업 브랜치 `feature/natural-face-expression`에서 진행한다. 안정 `main`과 `v0.1.0`을 덮어쓰지 않는다. 9절의 15개 테스트 기록은 안정 버전의 기록이며 업데이트 1의 테스트는 총 25개다.

### 구현 변경

- Source 누끼: 얼굴/헤어/피부/배경/의상/액세서리의 확률 지도와 얼굴 곡선 윤곽을 결합. guided filter와 안쪽 distance feather로 머리카락·배경 색이 경계 바깥으로 번지는 것을 줄임. 원본 알파 유지.
- `FaceAsset.triangles`는 더 이상 일부 기준점에 대한 Delaunay 인덱스가 아니다. MediaPipe 공식 해부학적 topology의 852개 삼각형이며 0~467번 랜드마크를 직접 참조한다. 반전·퇴화 삼각형은 건너뛴다.
- 매 프레임 눈꺼풀/입술/볼/턱/눈썹을 조밀한 메시로 변형. Optical flow에 맞춘 적응형 필터로 눈/입 변화의 지연을 줄이고 피부 흔들림을 완화한다. 10프레임마다 국소 얼굴 윤곽으로 CSRT 박스를 재보정한다.
- 원본 영상의 동공/눈 내부와 입안(치아·혀)은 보존한다. 업로드 이미지의 주변 눈꺼풀/입술/피부는 변형한다. 이 부분은 생성형 표정 재현과 구별해야 한다.
- Target 얼굴/헤어 분할로 헤어/비얼굴 영역을 보호하고, LAB 조명·피부색 보정, 시간적 색상 shift 필터 및 multiband blending을 적용한다.
- 작업 중간 코덱은 MKV/FFV1. 최종 출력은 기존 720p/30fps H.264/AAC 스테레오 계약 유지.
- 보고서: `compositor`, `mesh_vertices`, `mesh_triangles`, `source_hair_segmentation`, `target_hair_segmentation`, `mode_frames`, `expression_interiors`, `intermediate_codec` 추가. 영역 모드에서는 표정 처리를 수행했다고 기록하지 않는다.
- GUI 미리보기와 실제 렌더는 동일한 `composite(..., landmarker=...)` 경로를 사용한다. 렌더는 선택 지점별 `BlendState`로 색상 안정화 상태를 초기화한다.

### 추가 모델과 환경

새 Python 패키지는 추가하지 않았다. 기존 lock을 유지한다. 추가 모델은 `models/selfie_multiclass_256x256.tflite`:

- URL: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite
- SHA256: `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`
- 공식 클래스 순서: background, hair, body-skin, face-skin, clothes, others.
- `scripts/setup_local.py`가 두 모델을 다운로드·해시 검사하고 실제 로딩한다. 새 환경에서 `setup_local.bat` 실행.
- 근거: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter

### 확인 방법 및 남은 한계

- 2026-10-01 최종 `test_local_*.py` 25개 모두 통과(skip 없음, 11.320초). 두 모델 해시/로딩과 `pip check`, 새 모듈 구문 검사도 통과했다. 확대 비교 PNG를 확인했다.

- `tests/test_local_expression.py`의 10개 테스트 추가: 헤어 제거, 안쪽 페더, 투명도 유지, 색상 안정성, 고정 메시, 깜빡임 반응, 원본 눈/입안 보존, 입술/볼 변형, 닫힌 눈의 수치 안정성 등.
- 기존 테스트는 두 모델이 있어야 실행된다. skip을 통과로 기록하지 않는다.
- `scripts/preview_expression.py`는 공개 astronaut와 matplotlib의 grace_hopper 사진으로 비교 PNG를 생성한다. `workspace/expression-review/`에 누끼, 비교 이미지, 실행 정보가 남는다.
- 실제 사용자 영상과 단일 사진에서의 모든 3D 근육/가림/극단적 측면 재현은 검증되지 않았다. 합성된 표정 테스트는 실제 근육 움직임 영상의 대체물이 아니다.
- 영역 모드는 여전히 표정 전이를 하지 않는다. 완료 창에 정밀/영역 프레임 수를 각각 표시한다.
- 매 프레임 분할과 852개 삼각형 연산으로 CPU 처리 시간이 늘어난다. 실시간 처리 성능을 보장하지 않는다.

## 15. 실제 YouTube 10초 검증 (2026-10-01)

- 업데이트 1의 코드 `9de1c82`로 NASA 영상 `https://www.youtube.com/watch?v=u80H3FpTezA`의 34–44초 Jessica Meir 구간에 앞선 테스트의 `astronaut.png` 얼굴을 합성했다.
- 산출물: `outputs/youtube_quality_20261001/face_swap_10s.mp4`, `comparison_10s.mp4`, `quality_contact_sheet.jpg`, `quality_metrics.json`, `QUALITY_REPORT.ko.md`. 이 폴더는 Git에서 제외되므로 다른 환경에는 별도로 전달한다.
- 실행 도구: `workspace/youtube-qa/render_sample.py`, `quality_check.py`. 다운로드용 yt-dlp 2026.8.19는 `.runtime/qa-tools`에 격리했으며 제품 의존성을 변경하지 않았다.
- 720p/30fps H.264 MP4, AAC 48kHz 스테레오. 전체 300프레임 디코딩 정상. 300/300프레임 메시 합성, 추적 건너뜀/영역 대체 없음. 렌더 221.15초. 원본 대비 오디오 파형 측정 지연 0ms.
- 100개 샘플에서 눈 개방비 변화 상관 약 0.985, 입 약 0.979. 동일 모델 계열에 의한 측정으로 자연스러움 점수나 독립적 품질 검증이 아니다.
- **품질 판정: 추적과 파일 규격은 통과하나 자연스러움은 미달.** 초별 확대 비교에서 이마를 가로지르는 경계, 피부톤 차이, 2–3초 중립 표정의 미소 잔상, 6초 부근 눈꺼풀 흐림이 보였다. 다음 작업은 이마 매트/색상 경계와 표정 질감 보완을 우선한다. 회귀 테스트 통과를 실제 자연스러운 합성의 증거로 취급하지 않는다.
- 이번에는 엔진에 ROI 데이터를 직접 전달했다. GUI 드래그 자체, 극단적 측면/가림/여러 인물, 전 프레임의 미세한 떨림에 대한 육안 검증은 포함하지 않았다.

## 16. 교체 얼굴 특징 조절 (2026-10-01)

사용자가 요청한 닮음/움직임 균형 개선을 같은 `feature/natural-face-expression` 브랜치에서 진행했다. 새 모델이나 패키지는 추가하지 않았다.

### 인터페이스와 기본값

- `src/local/identity.py`의 불변 `CompositeSettings(identity, lighting, skin_color, detail)`를 `composite(..., settings=...)`, `render_video(..., settings=...)`로 전달한다. 설정 생략은 기존 계수 유지이며, GUI/CLI의 명시적인 기본 프리셋은 balanced다.
- 프리셋 순서: identity / lighting / skin_color / detail. legacy `(0, .65, .45, .25)`, balanced `(.35, .50, .25, .10)`, strong `(.55, .40, .15, .05)`.
- identity 범위 0–0.65, 나머지 0–1. NaN/무한대/범위 초과 거부. CLI `--preset` 뒤 개별 `--identity`, `--lighting`, `--skin-color`, `--detail`로 덮어쓸 수 있다.
- GUI 영상 아래 프리셋과 네 계수를 표시한다. 변경 후 선택 시점 미리보기와 실제 저장에 같은 설정 스냅샷을 사용한다. 영역 모드에서는 비율 조절을 비활성화하며 엔진에서도 무시한다.

### 합성 방식과 한계

- 선택 시점마다 `IdentityState`를 초기화한다. 눈 축과 얼굴 폭으로 source/driver를 정규화하고 눈 간격, 코 폭, 입 폭 차이를 추출한다. 눈/입 개방과 입꼬리 높이는 교체 사진에서 복사하지 않는다.
- 2D 공간에서 특징별 변형을 부드럽게 주변 피부로 연결하고 얼굴 외곽에서는 0으로 감쇠한다. 얼굴 폭 대비 변위 상한을 두며 유효한 삼각형의 반전·면적 붕괴가 발생하면 전체 안전 강도를 절반씩 줄인다.
- 프리셋별 강도 역전을 막기 위해 최대 0.65에서 공통 안전 한도를 구한 뒤 요청 강도를 비례 적용한다. 안전 한도 하락은 즉시 반영하고 회복은 0.1 계수로 완화한다. 실제 적용 강도는 요청값보다 상당히 낮을 수 있다.
- 눈 내부/입안도 원본 영상에서 새 위치로 변형한다. 타깃 헤어/비얼굴 보호 마스크를 이동한 내부 영역에도 적용한다. 색상 보정은 원본 프레임의 피부 기준을 사용한다.
- 보고서 엔진명 `dense-expression-v3`, `composite_settings`, `identity_geometry_frames`, `identity_reduced_frames`, `identity_reduction_steps`, `identity_applied_strength_min`, `identity_applied_strength_mean` 추가. 감소 프레임은 안전 회복 중인 프레임도 포함한다.
- 사진의 정적 미소 질감, 이마 경계 및 3D 자세 문제는 남는다. 코 폭 등은 2D 투영 비율이며 3D 신원 복원이나 학습 기반 identity transfer가 아니다. 첫 프레임의 표정/자세가 기준 비율에 영향을 줄 수 있다.
- 선택 시점 미리보기는 상태를 새로 초기화한다. 이후 프레임의 누적 색상/안전 강도 상태까지 미리보기와 같다고 주장하지 않는다.

### 검증과 재현

- 최종 엔진 기준 `test_local_*.py` **37개 통과, skip 없음, 41.444초**. 기존 동작 호환, 형태 강도, 사진의 미소/깜빡임 비복사, 눈·입안 이동, 헤어 보호, 반전 방지, 설정 유효성, GUI 미리보기/저장 설정 일치, 재선택과 결과 통계를 포함한다. CLI `--help`도 Windows 기본 문자 인코딩에서 확인했다.
- `scripts/compare_identity.py`: 정규화한 영상, 얼굴, selections를 받아 세 프리셋 MP4와 얼굴 기준 사진이 포함된 4열 비교 영상을 생성. `--preview-only`는 정적 미리보기만 생성. `--presets balanced strong`은 기존 legacy 결과를 보존하고 두 설정만 재생성한다.
- `scripts/check_identity_quality.py`: 프리셋별 전체 디코딩, 3프레임 간격 눈·입 개방비 변화, 원본 대비 오디오 지연, 실제 렌더 프레임 비교 이미지를 생성한다. 표정 상관계수 0.95를 회귀 기준으로 쓰며 닮음/자연스러움 점수로 해석하지 않는다.
- 입력은 15절과 동일. 산출물은 Git에서 제외되는 `outputs/identity_quality_20261001/`에 저장한다. 다른 환경에는 입력과 출력을 별도로 전달한다.
- 실제 세 프리셋 10초 검증 완료: 각각 300/300프레임 메시 합성, 추적 건너뜀 0, 720p/30fps H.264/AAC 스테레오, 전체 디코딩 정상, 오디오 지연 0ms. 눈/입 변화 상관은 legacy 0.979–0.985, balanced 0.984–0.988, strong 0.979–0.987로 모두 0.95 기준 통과.
- 기존 프리셋 결과와 15절의 이전 결과는 300프레임 디코딩 픽셀이 완전히 동일했다. 새 설정의 실제 비율 강도 평균은 balanced 0.1004, strong 0.1578(요청 0.35/0.55)이며 안전 제한/회복으로 전 프레임에서 요청값보다 낮았다. 형태 개선이 제한적임을 설명해야 한다.
- 렌더 시간 legacy 219.53초, balanced 267.86초, strong 266.41초. 확대 비교에서 교체 이미지의 색·질감은 더 남으나 이마 경계와 중립 표정의 미소 질감은 여전히 보였다. `QUALITY_REPORT.ko.md`에 한계와 실제 확인 범위를 명시했다. 결과를 자연스러운 신원 교체의 완성으로 주장하지 않는다.

```powershell
.venv\Scripts\python.exe scripts/compare_identity.py --video outputs/youtube_quality_20261001/source_10s.mp4 --face workspace/test-assets/astronaut.png --selections outputs/youtube_quality_20261001/selections.json --out-dir outputs/identity_quality_20261001
.venv\Scripts\python.exe scripts/check_identity_quality.py --source outputs/youtube_quality_20261001/source_10s.mp4 --folder outputs/identity_quality_20261001 --roi 450 100 380 400
```
