# Face Swap Pipeline

Deep fake와 비슷한 구조의 얼굴 교체 파이프라인입니다.

## 🏗️ 프로젝트 구조

```
face-swab2/
├─ configs/                  # YAML 설정
│  ├─ deepfacelab.yaml
│  ├─ fomm.yaml
│  └─ default.yaml
├─ data/
│  ├─ raw/                   # 원본(source, target)
│  └─ processed/             # 전처리 결과
├─ src/
│  ├─ common/                # 공통 유틸 (video_io, landmarks 등)
│  ├─ preprocess/            # 프레임 추출, 얼굴 검출/정렬
│  ├─ deepfacelab/           # DFL 모델 학습/추론
│  ├─ fomm/                  # FOMM 모델 학습/추론
│  ├─ postprocess/           # 색보정, temporal smoothing
│  ├─ audio/                 # 오디오 추출/정렬
│  └─ cli.py                 # 전체 파이프라인 오케스트레이터
├─ requirements.txt
└─ scripts/                  # 실행 스크립트
```

## 🔄 파이프라인 단계

### 1. Preprocess
- 영상 → 프레임 추출
- 얼굴 검출 + 랜드마크 + 정렬/alignment
- 마스크 + 메타데이터 저장

### 2. Training
- **DeepFaceLab**: encoder/decoder 기반 얼굴 교체
- **FOMM**: keypoint detector + generator 기반 모션 전달

### 3. Inference
- 학습된 모델로 얼굴/모션 변환 수행
- 프레임 시퀀스 생성

### 4. Postprocess
- 색상/조명 보정
- OpenCV seamlessClone (Poisson blending)
- temporal smoothing (artefact 최소화)

### 5. Audio
- 오디오 추출 (ffmpeg)
- 옵션: diarization, TTS, lip-sync

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# Windows
scripts\setup.bat

# 또는 수동 설치
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 실행 방법

#### 🎯 방법 1: GUI 모드 (권장)

##### 간단한 시작
```bash
# 더블클릭으로 실행 (3가지 방법)
실행.bat          # 한글명 실행 파일
얼굴교체.bat      # 간단한 실행 파일
start.bat         # 메뉴가 있는 실행 파일
```

##### GUI 사용법
1. **1단계: 얼굴 추출**
   - 소스 비디오 파일 선택
   - 샘플링 비율 설정 (1-10, 낮을수록 정확)
   - 출력 폴더 선택
   - "얼굴 추출 시작" 버튼 클릭

2. **2단계: 얼굴 교체**
   - 타겟 이미지 선택
   - 메타데이터 파일 자동 설정
   - 처리 모드 선택 (LightWarp/FOMM)
   - 출력 파일 선택
   - "얼굴 교체 시작" 버튼 클릭

3. **실행 로그**
   - 실시간 처리 상황 확인
   - 로그 저장 및 지우기 기능

##### 직접 GUI 실행
```bash
python face_swap_gui.py
```

#### 방법 2: 명령줄 모드
```bash
scripts\run.bat
```

#### 방법 2: 개별 스크립트

##### 얼굴 추출 (프롬프트 A)
```bash
# 비디오에서 얼굴 추출 및 트래킹
python scripts\extract_faces.py --video "source.mp4" --sample-rate 3 --out "data\processed\source"
```

##### 얼굴 교체 (프롬프트 B)
```bash
# LightWarp 모드 (빠른 기하학적 변환)
python scripts\swap_faces.py --mode lightwarp --target "target.jpg" --video "source.mp4" --meta "data\processed\source\source_video_meta.json" --out "outputs\result_lightwarp.mp4"

# FOMM 모드 (고품질 딥러닝)
python scripts\swap_faces.py --mode fomm --target "target.jpg" --video "source.mp4" --meta "data\processed\source\source_video_meta.json" --checkpoint "checkpoints\fomm.pth" --out "outputs\result_fomm.mp4"
```

##### 기존 파이프라인
```bash
# 학습 (FOMM)
python scripts\train.py --model "fomm" --data "data\processed" --output "checkpoints"

# 추론
python scripts\inference.py --model "fomm" --source "source.jpg" --target "target.mp4" --output "result.mp4" --checkpoint "checkpoints\best.pth"

# 후처리
python scripts\postprocess.py --input "result.mp4" --output "final_result.mp4"

# 오디오 처리
python scripts\audio.py --input "final_result.mp4" --output "final_with_audio.mp4" --enhancement
```

#### 방법 3: 전체 파이프라인
```bash
python scripts\run_full_pipeline.py --source "source.mp4" --target "target.mp4" --output "result.mp4" --model "fomm"
```

## 📋 사용 예시

### 새로운 얼굴 교체 파이프라인

#### 1단계: 얼굴 추출 (프롬프트 A)
```bash
# 비디오에서 얼굴 검출 및 트래킹
python scripts\extract_faces.py --video "source.mp4" --sample-rate 3 --out "data\processed\source"

# 결과:
# - data\processed\source\source_video_meta.json (메타데이터)
# - data\processed\source\face_tracks\<face_id>\frames\*.png (얼굴 크롭)
```

#### 2단계: 얼굴 교체 (프롬프트 B)

##### LightWarp 모드 (빠른 처리)
```bash
python scripts\swap_faces.py --mode lightwarp --target "target.jpg" --video "source.mp4" --meta "data\processed\source\source_video_meta.json" --out "outputs\result_lightwarp.mp4"
```

##### FOMM 모드 (고품질)
```bash
# FOMM 체크포인트 다운로드 필요
python scripts\swap_faces.py --mode fomm --target "target.jpg" --video "source.mp4" --meta "data\processed\source\source_video_meta.json" --checkpoint "checkpoints\fomm.pth" --out "outputs\result_fomm.mp4"
```

### 기존 파이프라인

#### DeepFaceLab 모델 사용
```bash
# 1. 전처리
python scripts\preprocess.py --source "person_a.mp4" --target "person_b.mp4" --output "data\processed"

# 2. 학습 (수동으로 데이터셋 정보 필요)
# 3. 추론
python scripts\inference.py --model "deepfacelab" --source "person_a.mp4" --target "person_b.mp4" --output "result.mp4" --checkpoint "checkpoints\deepfacelab\best.pth"
```

#### FOMM 모델 사용
```bash
# 1. 학습
python scripts\train.py --model "fomm" --data "target_video.mp4" --output "checkpoints"

# 2. 추론
python scripts\inference.py --model "fomm" --source "source_image.jpg" --target "target_video.mp4" --output "result.mp4" --checkpoint "checkpoints\best.pth"
```

## ⚙️ 설정

### 모델 설정
- `configs/deepfacelab.yaml`: DeepFaceLab 모델 설정
- `configs/fomm.yaml`: FOMM 모델 설정
- `configs/default.yaml`: 기본 설정

### 주요 설정 옵션
```yaml
preprocessing:
  face_detector: "mediapipe"  # mediapipe, dlib, opencv, mtcnn
  face_alignment: true
  sample_rate: 1

model:
  type: "deepfacelab"  # deepfacelab, fomm
  resolution: 256

training:
  batch_size: 8
  learning_rate: 5e-5
  epochs: 1000

postprocessing:
  color_correction:
    enabled: true
    method: "histogram_matching"
  temporal_smoothing:
    enabled: true
    window_size: 5
```

## 🔧 주요 기능

### 새로운 얼굴 교체 파이프라인

#### 얼굴 추출 (프롬프트 A)
- **MediaPipe 기반 얼굴 검출**: 468개 랜드마크 추출
- **다중 트래커 지원**: CSRT, KCF, MOSSE
- **얼굴 트래킹**: IoU 기반 매칭, 손실 얼굴 복구
- **품질 필터링**: 대칭성, 크기 기반 품질 점수
- **메타데이터 저장**: JSON 형식으로 프레임별 얼굴 정보 저장

#### 얼굴 교체 (프롬프트 B)
- **LightWarp 모드**:
  - Delaunay 삼각분할 기반 기하학적 변형
  - 히스토그램 매칭 색상 전달
  - Poisson 블렌딩
- **FOMM 모드**:
  - First Order Motion Model 기반 고품질 생성
  - 딥러닝 기반 얼굴 애니메이션
- **시간적 스무딩**: 마스크 일관성 유지
- **오디오 보존**: ffmpeg 기반 오디오 합성

### 기존 기능

#### 전처리
- 다양한 얼굴 검출기 지원 (MediaPipe, dlib, OpenCV, MTCNN)
- 얼굴 정렬 및 랜드마크 추출
- 품질 필터링 및 데이터 증강

#### 모델
- **DeepFaceLab**: SAE, LIAE, DF 아키텍처 지원
- **FOMM**: First Order Motion Model
- GAN 기반 학습 및 판별자

#### 후처리
- 색상 보정 (히스토그램 매칭, Reinhard, LCT)
- 시간적 스무딩
- Seamless 블렌딩 (Poisson, Mixed)
- 이미지 향상 (언샤프 마스크, CLAHE, 노이즈 감소)

#### 오디오
- 오디오 추출 및 동기화
- 립 싱크 처리
- TTS 지원
- 오디오 향상

## 📝 로그

로그는 `logs/face_swap.log`에 저장됩니다.

## 🐛 문제 해결

### 일반적인 문제
1. **CUDA 오류**: GPU 메모리 부족 시 배치 크기 줄이기
2. **메모리 부족**: 샘플링 비율 조정 (`--sample-rate`)
3. **얼굴 검출 실패**: 다른 검출기 시도 (`face_detector` 설정 변경)

### 성능 최적화
- GPU 사용 권장
- 배치 크기 조정
- 샘플링 비율 조정
- 해상도 조정

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로만 사용해주세요.

## 🤝 기여

버그 리포트나 기능 요청은 이슈로 등록해주세요.
