# faceswab2 코드 및 로컬 실행 사전 점검

점검일: 2026-09-30. 저장소: https://github.com/splendidhm/face-swab2.git
로컬: `C:\Users\케이지에프앤비\codex\projects\faceswab2`
기준: `main`, `33ed486` (2025-09-22, Initial commint : first). 복제 시 이력은 커밋 1개.

## 결론과 점검 범위

현재 저장소는 바로 사용할 수 있는 완성된 얼굴 합성 프로그램이 아니다. GUI는 합성을 시뮬레이션하고, CLI의 실제 합성 경로에도 프레임 누락 및 데이터 형상 불일치가 있다. 의존성 설치만으로 해결되지 않는다.

이번 작업은 저장소 복제, 소스 정적 검토, PC 환경 조회, 최신 공식 문서 조사이다. 실행 코드 변경, 패키지 설치, 모델 다운로드, 실제 영상 합성은 수행하지 않았다. 실행 가능한 Python이 PATH와 확인한 사용자 설치 폴더에서 발견되지 않아 단위 테스트와 런타임 재현은 하지 못했다. 아래 오류는 소스에서 확인한 사항이며 실행 테스트 결과와 구분한다.

## 전체 구조와 실제 진입점

| 경로 | 역할 및 상태 |
|---|---|
| `face_swap_gui.py` | Tkinter UI. 추출은 SimpleVideoPreprocessor 호출. 합성은 미연결 |
| `start.bat`, `run_gui.bat`, 한글 이름 bat | 실행 진입점. Python/가상환경 필요. 작업 디렉터리 기준 상대경로 주의 |
| `face-swab2.py` | 빈 파일이며 실행 진입점이 아님 |
| `scripts/extract_faces.py` → `src/preprocess/video_extractor.py` | Haar 기반 검출과 트래커를 사용하는 추출 CLI |
| `scripts/simple_extract.py` → `simple_video_extractor.py` | 샘플 프레임만 검출, 지속적인 얼굴 ID 추적 없음 |
| `scripts/swap_faces.py` → `src/infer/swap_infer.py` | LightWarp/FOMM 합성 CLI |
| `src/infer/video_synthesizer.py` | JSON 메타데이터를 읽어 프레임 합성 후 FFmpeg로 원본 오디오 결합 |
| `src/infer/lightwarp.py` | Delaunay 기하 변형 → 색상 전달 → Poisson 블렌딩 |
| `src/infer/fomm_inference.py` | 로컬 FOMM 모델 로딩 및 얼굴 크롭 추론 |
| `src/cli.py` | 별도의 전처리→학습→추론→후처리→오디오 파이프라인 |
| `src/deepfacelab/`, `src/fomm/` | 자체 PyTorch 모델·학습·추론 코드. 이름만으로 공식 구현/체크포인트 호환을 보장할 수 없음 |
| `src/postprocess/`, `src/audio/`, `src/common/` | 색상·이미지 보정, 오디오 처리, 영상 입출력 및 유틸리티 |
| `configs/` | YAML 설정. 경로마다 설정 전달 방식이 달라 통합 필요 |
| `tests/` | 검출 및 LightWarp 단위 테스트. 실제 영상/오디오 종단 테스트 없음 |
| `data/processed/test/`, `outputs/` | 예제 메타데이터/얼굴 크롭 및 258바이트 임시 MP4. 성공 결과물 근거로 사용할 수 없음 |

현재 실제 CLI 흐름: 영상 → 검출/추적 → 프레임별 JSON → 얼굴 이미지 변형/생성 → 블렌딩 → 임시 MP4 → 오디오 결합.
GUI는 이 흐름의 마지막 합성 구간을 호출하지 않는다. GUI/새 CLI/기존 학습 CLI의 세 경로를 정리해야 한다.

## 수정 우선순위

| 우선순위 | 근거 | 영향 및 필요한 작업 |
|---|---|---|
| P0 | `face_swap_gui.py:325`, `:339` | `run_swap()`은 `time.sleep(2)` 후 성공 표시. 실제 엔진 호출과 결과 파일 검증 연결 필요 |
| P0 | `src/preprocess/face_tracker.py`의 `update()` | detections가 빈 비검출 프레임에서 기존 트래커를 갱신/반환하는 루프가 없음. 모든 프레임의 추적 결과를 제공해야 함 |
| P0 | `simple_video_extractor.py`와 `video_synthesizer.py` | 샘플 프레임만 JSON에 기록하고, JSON 없는 프레임은 원본으로 출력. sample_rate=3이면 교체/원본이 번갈아 나올 수 있음 |
| P0 | `src/infer/lightwarp.py:170`, `:239` | 전체 프레임 크기의 warped 이미지를 얼굴 bbox 크기로 resize한 뒤 전체 프레임 좌표의 마스크로 seamlessClone 호출. 이미지·마스크의 크기와 좌표계 일치 필요 |
| P0 | `src/fomm/model.py:250`, `src/infer/fomm_inference.py:73`, `:117` | 모델은 4개 값의 tuple을 반환하지만 후처리는 Tensor로 보고 detach 호출. 인터페이스 불일치 |
| P0 | `src/infer/fomm_inference.py` | 얼굴 크롭과 전체 프레임 마스크의 블렌딩 좌표계도 불일치. 체크포인트 파일 없음, 공식 가중치와 자체 구조 호환 검증 없음 |
| P1 | `src/infer/video_synthesizer.py:200` | `_smooth_masks()`는 입력 frame을 그대로 반환. 시간적 스무딩 구현 없음 |
| P1 | `face_detector_simple.py`, `lightwarp.py` | Haar bbox 비율로 만든 5개 점을 랜드마크로 사용. 실제 눈·코·입 검출이 아니므로 회전/표정 대응 제한 |
| P1 | `src/infer/video_synthesizer.py:16` | LightWarp만 써도 FOMM을 import하여 torch가 필요. 엔진별 지연 import와 의존성 분리 필요 |
| P1 | `requirements.txt` | torch/TensorFlow/dlib/Jupyter 등을 일괄 설치. OpenCV 두 배포판을 동시에 명시, 상한/lock 없음. scipy 직접 import지만 직접 의존성으로 미명시 |
| P1 | `src/preprocess/face_detector_advanced.py:26` 등 | `mp.solutions` 구형 API 사용. 최신 MediaPipe와 호환된다고 가정하면 안 됨 |
| P1 | GUI worker와 경로 변수 | worker에서 Tk 변수/위젯/메시지박스 접근. UI 갱신 큐 필요. 추출 폴더와 출력 영상이 같은 output_path 변수 사용 |
| P1 | 영상/오디오 처리 | 입력 영상 열림·FPS·실제 프레임 수 검증, 무음 영상 처리, 임시파일 정리, FFmpeg 실패 표시, H.264 출력 및 오디오 싱크 검증 필요 |
| P2 | 저장소 관리 | `.gitignore`, 의존성 lock, 모델 출처/해시 관리 부재. pycache와 임시 산출물 추적 정리 필요 |

예제 메타데이터는 30fps, 총 90프레임, sample_rate=3, 기록된 프레임 30개이다. 프레임 누락 구조를 뒷받침하지만 실제 출력 영상 검증을 대체하지 않는다.

기존 테스트에도 구현과 기대값 불일치가 있다. 예를 들어 color_transfer는 bbox 크기로 결과를 바꾸지만 테스트는 입력 이미지 크기 유지를 기대한다. 무작위 이미지에서 얼굴 미검출로 원본을 반환해도 통과하는 테스트는 합성 성공을 보장하지 못한다.

## PC 환경 및 필요한 요소

| 항목 | 확인 결과 / 조치 |
|---|---|
| CPU | Intel Core i5-1240P |
| RAM | 16,908,406,784 bytes, 약 15.75 GiB |
| GPU | Intel Iris Xe Graphics, 드라이버 30.0.101.1298. 별도 Mirage 가상 디스플레이도 표시 |
| NVIDIA | 조회된 GPU 목록에 없음. nvidia-smi도 PATH에서 미발견. CUDA를 기본 환경으로 선택하지 않음 |
| 디스크 | C: 약 764.6 GB 여유(조회 시점) |
| Python / py / conda | PATH에서 미발견. 사용자 Programs/Python 아래 python.exe도 미발견. 다른 경로의 설치 여부는 미확인 |
| FFmpeg / ffprobe | PATH에서 미발견. Python wrapper 설치와 별도로 실행파일 필요 |
| Git | 2.51.0.windows.1, 저장소 복제 성공 |
| 모델 | 저장소 추적 파일에 .onnx/.pth/.pt 없음 |

필요한 구성: 격리 Python 환경, 최소 실행 의존성, FFmpeg/ffprobe, 검출·랜드마크·얼굴 임베딩·교체 모델, 선택적 가림 마스크/복원 모델, 로컬 모델 캐시와 무결성 확인, 실행 런처, 샘플 이미지/영상.

## 최근 방식 적용 제안

실행 기반은 사전학습 ONNX 모델을 사용하는 얼굴 교체 엔진으로 교체하는 방안을 권장한다. 현재 PC에서는 CPU를 기준 실행 경로로 확보하고 Intel GPU는 별도 벤치마크 후 활성화한다. 실제 속도나 실시간 처리 가능 여부는 아직 측정하지 않았다.

1. **우선 후보: FaceFusion 엔진 연동.** 이 프로젝트의 입력/출력 UI를 개선하면서 버전을 고정한 외부 엔진의 CLI를 어댑터로 호출하면, 모델 관리·마스크·영상 처리의 재구현을 줄일 수 있다. 현재 공식 릴리스 페이지의 최신 표시는 3.9.0이며 AlphaFace 256 추가를 명시한다. 모델 문서는 HyperSwap/INSwapper 등도 제공한다. 새 모델이 이 PC에서 더 빠르거나 더 좋다는 뜻은 아니므로 동일 영상 비교가 필요하다.
2. **대안: 자체 ONNX 어댑터.** 기존 VideoSynthesizer에 검출→정렬→신원 임베딩→교체→역변환→가림 마스크→합성을 구현한다. 제어 범위는 넓지만 검증·유지보수 작업이 증가한다.
3. **LightWarp는 비교용 보조 경로.** 좌표계와 프레임 처리부터 고친 뒤 경량 기준선으로 사용한다. 고품질 최신 신경망 교체 모델을 대체하지는 못한다.
4. **FOMM/DeepFaceLab 자체 학습은 후순위.** 현재 자체 구현의 호환성과 학습 데이터부터 검증해야 하므로 첫 로컬 실행 목표에 적합하지 않다.

FaceFusion 공식 설치 문서는 Python 3.12와 CPU/DirectML/OpenVINO 설치 경로를 제시한다. 채택한 릴리스에 맞춰 의존성을 고정해야 하며 이 저장소의 기존 requirements와 혼합하지 않는다. ONNX Runtime 공식 문서상 OpenVINO는 Intel CPU/GPU/NPU를 지원하지만 이 PC의 드라이버·모델 조합에서 가속 가능한지는 실제 세션 생성과 처리 시간으로 확인한다. DirectML은 계속 지원되지만 신규 개발 방향은 WinML로 이동했다고 명시되어 있다.

## 구현 순서 및 완료 조건

1. Python 3.12 기반 격리 환경과 FFmpeg/ffprobe를 준비하고 환경 진단 명령을 만든다. 버전 선택은 채택 엔진 릴리스와 재확인한다.
2. CPU용 사전학습 추론 엔진 및 모델을 고정하고 이미지 1장→이미지 1장 합성을 검증한다.
3. 5~10초 영상으로 모든 프레임 처리, 얼굴 선택/추적, 원본 오디오 보존, 무음 영상 처리를 검증한다.
4. GUI를 실제 엔진에 연결하고 진행률·취소·오류·결과 미리보기와 분리된 입출력 경로를 제공한다.
5. Intel 가속을 CPU 결과와 비교한다. 처리시간, 메모리, 얼굴 품질, 깜빡임을 기록하여 기본값을 정한다.
6. 한글·공백 경로, 얼굴 없음/여러 명, 빠른 움직임/가림, 긴 영상, 입력과 출력 동일 경로 방지 등을 확인한다.

완료 판정: GUI가 생성된 파일을 검증한 뒤에만 성공 표시, 프레임 누락 없는 출력, 의도한 얼굴만 교체, 재생 가능한 영상/오디오, FFprobe 기준 길이·FPS·오디오 스트림 확인, 재설치 가능한 버전 고정 환경. 현재 단계에서는 이 완료 조건을 달성한 상태가 아니다.

## 공식 참고 자료

- FaceFusion 릴리스: https://github.com/facefusion/facefusion/releases
- 모델 옵션: https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper
- 설치: https://docs.facefusion.io/installation
- ONNX Runtime OpenVINO: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
- ONNX Runtime DirectML: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- MediaPipe 구형 API 제거 안내(공식 저장소 이슈): https://github.com/google-ai-edge/mediapipe/issues/6192

모델 선정 시 배포 코드와 가중치 각각의 사용 조건 및 배포 출처를 함께 기록해야 한다. 저장소에는 별도 LICENSE 파일이 없고 README는 교육/연구 목적 사용을 명시한다.
