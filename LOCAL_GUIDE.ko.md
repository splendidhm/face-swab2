# 설치 및 사용 안내

현재 기능 브랜치: `feature/natural-face-expression` (업데이트 1, 2026-10-01). 안정 기준은 `main`의 `v0.1.0`입니다.

## 이 PC에서 실행

프로젝트 폴더: `C:\Users\케이지에프앤비\codex\projects\faceswab2`

`run_gui.bat`, `start.bat`, `실행.bat`, `얼굴교체.bat`는 모두 같은 새 GUI를 실행합니다. 어느 작업 디렉터리에서 실행해도 프로젝트 폴더를 기준으로 동작합니다.

현재 설치 구성:

- `.runtime/python`: 서명이 검증된 python.org Python 3.12.10, Tcl/Tk 포함
- `.venv`: 프로젝트 전용 가상환경
- MediaPipe 0.10.32 Tasks API, OpenCV contrib 4.11.0.86
- `models/face_landmarker.task`: Google 공식 모델 v1
- `models/selfie_multiclass_256x256.tflite`: 얼굴/헤어/피부/의상 분할 모델 v1. 기존 환경도 `setup_local.bat`로 추가 설치합니다.
- imageio-ffmpeg에 포함된 FFmpeg 7.1 실행파일
- `requirements-local.lock.txt`: 전체 의존성 고정

환경 재검사:

```powershell
.\.venv\Scripts\python.exe scripts\setup_local.py
.\.venv\Scripts\python.exe -m pip check
```

다른 PC는 Python 3.12와 Tk를 설치하고 `setup_local.bat`를 실행합니다. 설치 단계에만 인터넷이 필요합니다. 영상과 얼굴 이미지는 외부 서버로 전송하지 않으며 추론과 합성은 로컬 CPU에서 처리됩니다. CUDA/torch/TensorFlow 패키지는 새 앱의 의존성이 아닙니다.

## 입력과 얼굴 추출

- 영상: MP4, MOV, MKV, AVI, WebM 등 FFmpeg가 읽을 수 있는 형식. 불러오면 고정 30fps, 1280×720로 변환합니다. 세로/비표준 비율 영상은 잘라내지 않고 여백을 추가합니다.
- 얼굴 사진: 실제 JPEG 또는 PNG. EXIF 회전을 반영합니다. 정면 얼굴이 한 명 있어야 하며 여러 명이면 다시 선택하도록 안내합니다.
- 정면 판정: 양 눈과 코의 상대 위치를 검사하는 기하학적 필터입니다. 완전한 3D 자세 판정은 아니므로 선명한 정면 사진을 사용하세요.
- 누끼: 턱·볼·이마 윤곽을 곡선으로 연결하고 분할 모델로 헤어·배경·목 등을 제외합니다. 이미지 경계에 맞춘 필터와 안쪽 페더링으로 경계 바깥으로 색이 번지는 것을 줄입니다. 기존 PNG 알파도 반영합니다. 머리카락에 가려진 이마를 새로 만들어 채우지는 않습니다.
- 누끼 결과는 바둑판 배경에 미리보기하며 별도 투명 PNG로 저장할 수 있습니다.

## 영상 속 얼굴 선택

1. 영상 준비가 끝나면 슬라이더나 이전/다음 프레임 버튼으로 원하는 시점을 엽니다.
2. 얼굴의 이마부터 턱까지 드래그하여 감쌉니다. 표시 좌표는 실제 720p 영상 좌표로 변환됩니다. 역방향 드래그도 가능합니다.
3. 얼굴 이미지를 이미 가져왔다면 해당 시점의 합성 미리보기가 표시됩니다.
4. 다른 시점에 드래그하면 선택 지점이 추가됩니다. 같은 시점은 기존 선택을 교체합니다.
5. 출력할 때 각 선택 지점에서 추적을 다시 시작합니다. 첫 선택보다 앞선 구간은 원본입니다.

사람형 얼굴은 전체 얼굴 메시의 468개 점과 고정 852개 삼각형으로 누끼를 변형합니다. 눈꺼풀·입술의 급격한 변화에는 빠르게 반응하고 피부 위치의 작은 흔들림은 완화합니다. 눈동자와 입안의 치아·혀는 실제 영상에서 보존합니다. 볼·턱·눈썹 주변의 위치와 일부 조명 변화도 합성 얼굴에 전달하며 LAB 색상 보정과 다중 해상도 블렌딩으로 경계를 맞춥니다.

얼굴을 인식하지 못하는 그림/캐릭터는 선택 사각형을 CSRT로 추적하여 누끼를 리사이즈해 합성합니다. 영역 모드에는 표정 전이 또는 회전 추정이 없습니다. 완료 창과 보고서에 두 모드의 프레임 수를 표시합니다.

얼굴 추적이 실패하거나 급격한 장면 전환을 감지하면 다른 인물을 전역 검색하여 임의로 선택하지 않습니다. 해당 구간은 원본으로 남기고 결과 보고서에 프레임 범위를 기록합니다. 짧은 랜드마크 미검출은 같은 추적 영역에서 재검출하고, 지속 실패는 다음 수동 선택까지 중단합니다. 비슷한 얼굴의 교차·긴 가림에 대한 신원 보장은 없으므로 실제 영상 미리보기 및 결과 확인이 필요합니다.

## 출력

- 해상도: 1280×720, 픽셀 비율 1:1, 30fps
- 비디오: MP4 컨테이너, H.264, yuv420p, CRF 20, faststart
- 오디오: AAC, 48kHz, 192kbps, 2채널 스테레오
- 원본 오디오가 모노이면 스테레오로 변환하며, 무음이면 무음 스테레오 트랙 생성
- 원본보다 오디오가 짧으면 무음으로 채우고 출력 영상 길이에 맞춰 종료
- 출력 파일 옆 `.report.json`: 합성 프레임 수, 첫 선택 이전 프레임 수, 추적 실패 구간, 선택 좌표, 사용 모드
- `mode_frames`: `mesh`(정밀 표정) / `region`(영역 추적) 별 처리 프레임 수. 분할 사용 여부, 메시 개수, 중간 코덱 정보도 기록합니다.
- 중간 영상은 MKV/FFV1 무손실로 저장한 뒤 최종 H.264로 압축합니다. 임시 디스크 사용량은 이전 버전보다 늘어날 수 있습니다.

임시 합성 파일은 작업별 임시 폴더에서 처리하고 취소/실패 시 정리합니다. 완성 파일 검증 후 지정한 출력 경로로 교체합니다. 입력 영상을 출력으로 덮어쓰는 것은 차단합니다. `workspace/`의 가져온 영상과 누끼 캐시는 재사용/확인을 위해 남습니다. 앱 종료 후 필요 없는 파일은 삭제할 수 있습니다.

## CLI

좌표는 가져오기 변환 후 1280×720 영상 기준, 프레임은 30fps 기준입니다.

```powershell
.\.venv\Scripts\python.exe scripts\local_swap.py --video "input.mp4" --face "face.png" --roi 400 180 180 220 --start-frame 0 --out "outputs\result.mp4"
```

강제 영역 모드는 `--region-mode`를 추가합니다. GUI에서 저장한 보고서를 `--selections result.report.json`으로 전달하면 여러 선택 시점을 재사용합니다(`--roi` 대신 사용).

## 검증 결과

업데이트 1은 기존 회귀 테스트에 헤어 제거, 배경 번짐, PNG 투명도, 눈/입안 보존, 깜빡임, 입술/볼 변형, 삼각형 안정성, 시간적 색상 보정 테스트 10개를 추가했습니다. `test_local_*.py` 기준 총 25개이며, 실행에는 두 모델과 공개 샘플이 필요합니다.

2026-10-01 최종 실행에서 25개 모두 통과했고(skip 없음), 두 모델 해시/로딩 및 Python 의존성 검사도 통과했습니다. 공개 샘플의 확대 비교 이미지로 누끼·합성 경계를 확인했습니다.

2026-09-30 이 PC에서 아래 15개 테스트 통과(영상/모델 12개, GUI 3개):

- 실제 Tasks 모델로 정면 얼굴 누끼 생성, PNG 투명도 및 JPEG 처리
- 얼굴 없는 이미지 거부
- 선택한 얼굴만 변형하고 화면의 다른 얼굴 보존
- 움직이는 선택 얼굴 추적
- 장면 전환 후 다른 얼굴로 넘어가지 않고 추적 중단
- 영역 추적 모드
- 선택 전 구간을 유지하면서 720p H.264/AAC 스테레오 출력 및 전체 디코딩
- 모노 원본의 유음 스테레오 변환
- 취소 및 입력 파일 덮어쓰기 차단
- 세로 영상의 비율 유지 및 720p 여백 처리
- 장면 전환 후 새 선택 시점에서 추적 재개
- 얼굴 모델이 인식하지 못하는 그림의 영역 추적 전환
- GUI 역방향 드래그의 영상 좌표 변환과 여러 선택 시점
- 너무 작은 선택 차단
- worker 결과를 메인 UI 큐에서 처리

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_local_*.py" -v
```

테스트는 `workspace/test-assets/astronaut.png` 공개 샘플에서 만든 짧은 이동 영상으로 수행했습니다. 실제 사용자가 준비한 캐릭터 영상의 품질·추적 성능은 아직 검증하지 않았습니다. GUI는 숨긴 Tk 창을 이용한 코드 수준 통합 테스트를 수행했으며 수동 사용성 검증을 대체하지 않습니다. 이전 tests/test_lightwarp.py 등은 보관된 구형 엔진 테스트입니다.

공개 샘플: https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png
이 샘플은 테스트 환경에만 다운로드하며 Git에 포함하지 않습니다. 파일이 없으면 실제 모델 통합 테스트는 skip됩니다.

서로 다른 공개 인물 사진으로 누끼·합성 비교 이미지 생성:

```powershell
.\.venv\Scripts\python.exe scripts\preview_expression.py
```

결과는 `workspace/expression-review/comparison.png`이며, source는 astronaut, target은 설치된 matplotlib의 grace_hopper 샘플입니다. 정지 이미지의 경계 확인용이므로 실제 동영상 표정 품질을 보장하는 자료가 아닙니다.

모델: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
API 참고: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

얼굴/헤어 분할 모델: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite
SHA256: `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`
분할 클래스 문서: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter

## 다음 품질 개선 범위

현재 구현은 얼굴/헤어 분할과 촘촘한 메시 변형을 결합한 누끼 기반 합성이다. 신경망으로 3D 근육을 재구성하거나 숨겨진 피부를 생성하지 않으며, 신원 추적·손/물체 가림 전용 모델은 없다. 고개를 크게 돌리는 장면·가림·고도로 양식화된 캐릭터에서의 품질은 별도 엔진 또는 추가 검증이 필요하다. Intel GPU 가속은 적용하지 않았으며 CPU로 동작한다. 매 프레임 분할과 촘촘한 메시 처리로 기존 버전보다 처리 시간이 늘어난다.
