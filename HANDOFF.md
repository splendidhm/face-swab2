# FaceSwab2 세션 인수인계

기준: 2026-10-01. 경로는 프로젝트 루트 기준. 상세 이력은 `AGENT.md`, 작업 규칙은 `AGENTS.md`.

## 1. 프로젝트 목적

JPEG/PNG 정면 얼굴을 추출해 영상에서 드래그한 얼굴에 합성·추적하는 Windows 로컬 앱. 표정·움직임 유지, 720p MP4·스테레오 출력이 목표. 현재는 **2D 누끼/메시 합성**이며 생성형 신원 교체 엔진이 아니다.

## 2. 구현 완료

- 정면 얼굴 한 명 검출, 헤어/배경/목 제외 누끼·투명 PNG 저장, 경계 페더링.
- 영상 정규화, 프레임 탐색, 드래그 ROI, 선택 시점별 재추적, 미리보기·진행률·취소·저장.
- 468점/852삼각형 표정 변형, CSRT+광학 흐름+재검출, 색·조명 보정. 눈 내부/입안은 영상에서 보존·이동.
- 얼굴 특징 프리셋 기존/균형/특징 강화 및 비율·밝기·피부색·명암 계수.
- 오른쪽 마스킹 슬라이더/버튼/휠: **low 25%, midium 50%, strong 75%, very strong 100%**. 기본 100%. 미리보기 자동 갱신, 렌더 중 다음 프레임부터 변경·보고서 기록.
- CLI, 설치/모델 검증, 회귀 테스트, 비교 영상·품질 측정 도구.

## 3. 파일 구조

- 루트: `face_swap_gui.py`, `face-swab2.py` 진입점, `run_gui.bat`, `setup_local.bat`, 안내 문서/requirements.
- `src/local/`: 현재 앱 엔진. `scripts/`: 실행·설치·비교 도구. `tests/test_local_*.py`: 현재 엔진 테스트.
- `.runtime/`, `.venv/`, `models/`, `workspace/`, `outputs/`: 로컬 전용, Git 제외.
- `src/infer/`, `src/fomm/`, `src/deepfacelab/` 등: 보관된 구형 엔진. 현재 GUI에서 사용하지 않음.

## 4. 주요 파일·함수

| 경로 | 역할/확인할 함수 |
|---|---|
| `src/local/gui.py` | `FaceSwapGUI`: `mask_changed`, `mask_wheel`, `mask_preview`, `refresh_preview`, `export`, `poll`, `close` |
| `src/local/faces.py` | `Landmarker`, `FaceAsset`, `extract_face`, `composite`: 검출·누끼·최종 합성 |
| `src/local/matting.py` | `FaceSegmenter`, `face_matte`, `inward_feather`: 피부/헤어 분할·경계 |
| `src/local/expression.py` | `dense_warp`, `aperture_mask`, `match_lighting`, `multiband_blend`, `BlendState` |
| `src/local/identity.py` | `CompositeSettings.preset`, `IdentityState.deform`, `safe_geometry`: 얼굴 비율·왜곡 방지 |
| `src/local/masking.py` | `MASK_LEVELS`, `MASK_STRENGTHS`, `MaskStrengthControl.set_level/snapshot`: 스레드 안전 강도 공유 |
| `src/local/tracking.py` | `SelectedTracker.update/current`, `face_in_region`: 선택 얼굴 추적 |
| `src/local/pipeline.py` | `render_video`: 프레임 처리, 상태 초기화, 강도 변경 이력, 저장/검증 |
| `src/local/media.py` | `normalize_video`, `read_frame`, `export_mp4`, `run_ffmpeg`: 입출력 |
| `scripts/local_swap.py`, `scripts/setup_local.py` | CLI / 모델 다운로드·SHA256·실행 환경 검사 |
| `scripts/compare_identity.py`, `scripts/check_identity_quality.py` | 3개 특징 프리셋 비교 / 표정·오디오·디코딩 검증 |

## 5. 실행 방법

- 작업 폴더: `C:\Users\케이지에프앤비\codex\projects\faceswab2`.
- 현재 PC: `run_gui.bat` 실행. `실행.bat` 등도 같은 런처에 연결됨. 새 코드 반영은 앱 재시작.
- 새 PC: Python 3.12 + Tk 설치 → `setup_local.bat` → `run_gui.bat`. 기존 가상환경을 복사하지 말 것.
- CLI 안내: `.venv\Scripts\python.exe scripts/local_swap.py --help`. 필수 `--video`, `--face`, `--out`, `--roi` 또는 `--selections`.
- 옵션: `--preset legacy|balanced|strong`, `--mask-level low|midium|strong|"very strong"`. PowerShell 한글 출력은 `$env:PYTHONUTF8='1'` 권장.

## 6. 주요 의존성

Python 3.12.10 / Tk 8.6, NumPy 2.2.6, opencv-contrib-python 4.11.0.86, MediaPipe 0.10.32, Pillow 11.3.0, SciPy 1.15.3, imageio-ffmpeg 0.6.0(FFmpeg 7.1). 정확한 설치 목록: `requirements-local.lock.txt`.

모델: `models/face_landmarker.task`, `models/selfie_multiclass_256x256.tflite`; 설치 도구가 다운로드/해시 검사. OpenCV는 **contrib** 필요. 새 GUI에 torch/TensorFlow/dlib/FaceFusion/ONNX Runtime 의존성 없음. yt-dlp 2026.8.19는 QA 전용 `.runtime/qa-tools`에 격리.

## 7. 테스트 결과

- 최종 전체: **44개 통과, skip 0, 29.488초**. 이후 안내 문구 복구 수정에 GUI 7개 재검증 통과.
- 명령: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_local_*.py" -v`.
- 테스트 입력 `workspace/test-assets/astronaut.png`와 두 모델 필요. 누락에 따른 skip을 성공으로 기록하지 말 것.
- 이전 3프리셋 10초 영상: 각각 300/300프레임 합성, 추적 손실 0, 전체 디코딩 정상, 오디오 지연 0ms. 눈/입 개방비 상관 0.979–0.988. **닮음 점수가 아님**.
- 슬라이더: 실제 짧은 렌더 중 변경·적용 프레임 기록 검증, 네 단계 이미지 육안 확인. 새 슬라이더로 10초 전체를 다시 렌더한 검증은 아님.

## 8. 발생 오류·경고

- GitHub 샌드박스 네트워크 연결 실패 → 승인된 외부 네트워크 실행으로 push 성공.
- 잘못된 GitHub 계정 `Hidden-Patterns-Lab` 접근 실패 이력. 이 저장소는 **splendidhm** 인증 사용.
- Windows cp949 출력 깨짐/CLI 도움말 en dash 인코딩 오류 → UTF-8 런처, 도움말 ASCII 처리.
- Tk 종료 후 예약 콜백 오류 → 예약 취소 처리. 초기 형태 강도 역전/과도한 제한 → 공통 안전 한도·공간 보간 수정.
- 남는 비차단 경고: Git LF→CRLF, MediaPipe feedback tensors/single-signature 안내. 테스트·출력 실패로 이어지지 않았음.

## 9. 해결한 문제

가짜 합성 대기를 실제 엔진으로 교체; 헤어 분리·표정 메시·색상 안정화; 이동 눈/입의 헤어 침범 방지; 강도 감소 시 색 보정이 꺼지는 문제 방지; GUI/CLI 설정 전달 통일; 렌더 중 Tk 접근 없이 강도 변경; 인코딩 중 조작 잠금·완료 복구; Git 계정/브랜치 분리.

## 10. 미해결·한계

- 이마 가로 경계, 피부톤 접합, 정적 사진의 미소/볼 질감 잔상, 눈 감김 흐림.
- 닮음 개선은 제한적. 안전 제한 후 평균 형태 강도 balanced 0.1004 / strong 0.1578(요청 0.35/0.55).
- 3D 복원·가려진 피부 생성 없음. 측면/손 가림/여러 인물/빠른 동작·전 프레임 미세 떨림 미검증. 영역 모드는 표정 변형 없음.
- CPU에서 10초 렌더 약 220–268초. 실시간 아님. 긴 영상 FFV1 디스크 사용량 주의.
- 숨긴 Tk 기능/크기 검증 완료이나 다양한 화면 배율·작은 화면의 수동 사용성 검증은 남음.

## 11. 다음 작업 우선순위

1. 새 세션에서 브랜치/미커밋 상태 확인, 기존 비교 이미지·품질 보고서부터 읽기. 새 PC이면 환경/로컬 입력 복원 후 테스트.
2. 실제 GUI에서 드래그·휠·4단계·렌더 중 변경·취소·완료 복구를 사용자 화면 배율로 확인.
3. 이마 누끼 접합과 피부색 경계 개선. 동일 10초 입력으로 전후 비교.
4. 미소 잔상·눈 감김·안전 제한에 따른 닮음 약화 개선. 요구 품질이 현재 2D 방식 한계를 넘으면 별도 엔진 평가.
5. 가림/측면/빠른 동작/다중 인물·긴 영상 성능 검증. 안정화 후 사용자 지시에 따라 PR/병합.

## 12. 유지할 설계 결정

- 안정 `main`/`v0.1.0` 보존. 새 기능은 `feature/*`, 수정은 `fix/*`; 임의 병합/태그 이동 금지.
- GUI/CLI는 같은 `composite`·`render_video` 사용. Tk는 메인 스레드, worker는 큐와 `MaskStrengthControl`만 사용.
- 출력 1280×720/30fps/H.264 yuv420p MP4 + AAC 48kHz 스테레오; 중간 FFV1. 입력 보존, 추적 실패 시 원본 유지·기록.
- ROI는 정규화된 720p 좌표, 프레임은 0부터. 여러 선택은 재추적 지점이지 여러 얼굴 동시 교체가 아님.
- 얼굴 특징 프리셋과 마스킹 불투명도는 독립. GUI/CLI 특징 기본 balanced, Python 설정 생략은 legacy. 마스킹 기본 very strong=1.
- 마스킹은 완성된 합성 결과와 원본의 혼합. 렌더 중 변경은 다음 처리 프레임부터; `masking_strength_events` 기록. 과거 프레임 소급 변경 없음.
- 모델/가상환경/개인 입력·출력은 Git 제외. 통과한 테스트나 표정 상관계수를 자연스러움/닮음 보증으로 해석하지 말 것.

## 13. 재조사 불필요한 정보

- 저장소: https://github.com/splendidhm/face-swab2.git. 현재 `feature/masking-strength-slider`, 최신 기능 커밋 **`6482b56`** (`마스킹 강도 슬라이드바 추가`), 원격 반영 완료. 이후 문서 커밋은 `git log -1`로 확인.
- 부모 `00e9fcf` (`얼굴 누끼 추적 강화`)는 `feature/natural-face-expression`에 push됨. 두 기능 브랜치는 main 미병합. 안정 `v0.1.0`/main은 `3240b28`.
- GCM의 splendidhm 인증 및 저장소 설정 `credential.https://github.com.username=splendidhm` 사용. 전역 gh/커넥터 계정에 의존하지 말 것. 비밀 토큰 출력 금지.
- PC: i5-1240P, RAM 약 16GB, Intel Iris Xe. NVIDIA/CUDA 없음. MediaPipe Tasks 사용(`mp.solutions` 아님), 한글 모델 경로는 바이트 로딩으로 처리.
- QA 원본: NASA https://www.youtube.com/watch?v=u80H3FpTezA 의 34–44초. 교체 사진은 astronaut 테스트 이미지이며 사용자 개인 사진이 아님.
- 로컬 자료: `outputs/youtube_quality_20261001/`, `outputs/identity_quality_20261001/`의 `QUALITY_REPORT.ko.md`, `comparison.mp4`, JSON; `outputs/masking_strength_20261001/levels.jpg`.
- 정규화 원본: `outputs/youtube_quality_20261001/source_10s.mp4`, 선택값 `selections.json`. 재다운로드 불필요. 재현 명령은 `AGENT.md` 16절.
- 작성 직전 코드 작업 트리는 깨끗했음. 이 문서는 별도 **`세션 변경`** 커밋으로 관리한다. 다른 PC에서는 해당 기능 브랜치를 가져오고 Git 제외 입력/출력은 별도로 전달할 것.
