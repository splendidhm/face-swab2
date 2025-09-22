"""
얼굴 랜드마크 검출 및 처리 유틸리티
"""
import cv2
import numpy as np
import mediapipe as mp
import face_alignment
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class LandmarkDetector:
    """얼굴 랜드마크 검출 클래스"""
    
    def __init__(self, method: str = 'mediapipe'):
        """
        Args:
            method: 검출 방법 ('mediapipe', 'face_alignment')
        """
        self.method = method
        
        if method == 'mediapipe':
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        elif method == 'face_alignment':
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False, 
                device='cpu'
            )
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        이미지에서 얼굴 랜드마크 검출
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            landmarks: 랜드마크 좌표 (N, 2) 또는 None
        """
        if self.method == 'mediapipe':
            return self._detect_mediapipe(image)
        elif self.method == 'face_alignment':
            return self._detect_face_alignment(image)
    
    def _detect_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe를 사용한 랜드마크 검출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # 첫 번째 얼굴의 랜드마크 추출
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = []
        h, w = image.shape[:2]
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def _detect_face_alignment(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Face Alignment를 사용한 랜드마크 검출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.fa.get_landmarks(rgb_image)
        
        if landmarks is None or len(landmarks) == 0:
            return None
        
        return landmarks[0]  # 첫 번째 얼굴
    
    def get_face_bbox(self, landmarks: np.ndarray, margin: float = 0.2) -> Tuple[int, int, int, int]:
        """
        랜드마크로부터 얼굴 바운딩 박스 계산
        
        Args:
            landmarks: 랜드마크 좌표
            margin: 여백 비율
            
        Returns:
            bbox: (x1, y1, x2, y2)
        """
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        margin_x = width * margin
        margin_y = height * margin
        
        x1 = max(0, int(x_min - margin_x))
        y1 = max(0, int(y_min - margin_y))
        x2 = int(x_max + margin_x)
        y2 = int(y_max + margin_y)
        
        return x1, y1, x2, y2
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                  output_size: int = 256) -> np.ndarray:
        """
        랜드마크를 사용하여 얼굴 정렬
        
        Args:
            image: 입력 이미지
            landmarks: 랜드마크 좌표
            output_size: 출력 크기
            
        Returns:
            aligned_face: 정렬된 얼굴 이미지
        """
        # 기준 랜드마크 (정면 얼굴)
        reference_landmarks = self._get_reference_landmarks(output_size)
        
        # 호모그래피 행렬 계산
        h, _ = cv2.findHomography(landmarks, reference_landmarks)
        
        # 이미지 변환
        aligned_face = cv2.warpPerspective(image, h, (output_size, output_size))
        
        return aligned_face
    
    def _get_reference_landmarks(self, size: int) -> np.ndarray:
        """기준 랜드마크 생성 (정면 얼굴)"""
        # 간단한 정면 얼굴 랜드마크 (실제로는 더 정확한 기준점 사용)
        center = size // 2
        landmarks = np.array([
            [center - 50, center - 30],  # 왼쪽 눈
            [center + 50, center - 30],  # 오른쪽 눈
            [center, center + 20],       # 코
            [center - 40, center + 60],  # 왼쪽 입
            [center + 40, center + 60], # 오른쪽 입
        ])
        return landmarks
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray, 
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        랜드마크를 이미지에 시각화
        
        Args:
            image: 입력 이미지
            landmarks: 랜드마크 좌표
            color: 색상 (BGR)
            
        Returns:
            vis_image: 시각화된 이미지
        """
        vis_image = image.copy()
        
        for point in landmarks:
            cv2.circle(vis_image, tuple(point.astype(int)), 2, color, -1)
        
        return vis_image


class FaceAligner:
    """얼굴 정렬 클래스"""
    
    def __init__(self, detector: LandmarkDetector):
        self.detector = detector
    
    def align_faces_in_video(self, video_path: str, output_dir: str, 
                           sample_rate: int = 1) -> List[Dict]:
        """
        비디오에서 얼굴 정렬 수행
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            sample_rate: 샘플링 비율
            
        Returns:
            metadata: 정렬된 얼굴 메타데이터
        """
        from .video_io import VideoIO
        
        video_io = VideoIO()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for frame, frame_idx in video_io.read_video_generator(video_path, sample_rate):
            landmarks = self.detector.detect_landmarks(frame)
            
            if landmarks is not None:
                aligned_face = self.detector.align_face(frame, landmarks)
                
                # 정렬된 얼굴 저장
                face_filename = f"aligned_face_{frame_idx:06d}.jpg"
                face_path = output_path / face_filename
                cv2.imwrite(str(face_path), aligned_face)
                
                # 메타데이터 저장
                metadata.append({
                    'frame_idx': frame_idx,
                    'face_path': str(face_path),
                    'landmarks': landmarks.tolist(),
                    'bbox': self.detector.get_face_bbox(landmarks)
                })
        
        return metadata

