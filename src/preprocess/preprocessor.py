"""
전처리 모듈 - 프레임 추출, 얼굴 검출, 정렬
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from tqdm import tqdm

from ..common.video_io import VideoIO
from ..common.landmarks import LandmarkDetector, FaceAligner
from ..common.utils import setup_logging, ensure_dir, save_metadata, get_timestamp


class Preprocessor:
    """전처리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = setup_logging()
        
        # 컴포넌트 초기화
        self.video_io = VideoIO()
        self.landmark_detector = LandmarkDetector(
            method=config.get('face_detector', 'mediapipe')
        )
        self.face_aligner = FaceAligner(self.landmark_detector)
    
    def extract_from_video(self, video_path: str, sample_rate: int = 1) -> List[Dict]:
        """
        비디오에서 프레임 추출 및 얼굴 검출
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 샘플링 비율
            
        Returns:
            extracted_data: 추출된 데이터 리스트
        """
        self.logger.info(f"비디오에서 프레임 추출 시작: {video_path}")
        
        extracted_data = []
        
        for frame, frame_idx in tqdm(
            self.video_io.read_video_generator(video_path, sample_rate),
            desc="프레임 추출 중"
        ):
            # 얼굴 검출
            landmarks = self.landmark_detector.detect_landmarks(frame)
            
            if landmarks is not None:
                # 얼굴 바운딩 박스 계산
                bbox = self.landmark_detector.get_face_bbox(landmarks)
                
                # 얼굴 품질 점수 계산
                quality_score = self._calculate_face_quality(frame, landmarks)
                
                # 데이터 저장
                frame_data = {
                    'frame_idx': frame_idx,
                    'landmarks': landmarks.tolist(),
                    'bbox': bbox,
                    'quality_score': quality_score,
                    'timestamp': frame_idx / self.video_io.get_video_info(video_path)['fps']
                }
                
                extracted_data.append(frame_data)
        
        self.logger.info(f"총 {len(extracted_data)}개의 얼굴 검출됨")
        return extracted_data
    
    def align_faces(self, video_path: str, output_dir: str, 
                   sample_rate: int = 1) -> List[Dict]:
        """
        비디오에서 얼굴 정렬 수행
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            sample_rate: 샘플링 비율
            
        Returns:
            aligned_metadata: 정렬된 얼굴 메타데이터
        """
        self.logger.info(f"얼굴 정렬 시작: {video_path}")
        
        ensure_dir(output_dir)
        aligned_metadata = self.face_aligner.align_faces_in_video(
            video_path, output_dir, sample_rate
        )
        
        # 메타데이터 저장
        metadata_path = Path(output_dir) / "aligned_metadata.json"
        save_metadata(aligned_metadata, str(metadata_path))
        
        self.logger.info(f"얼굴 정렬 완료: {len(aligned_metadata)}개")
        return aligned_metadata
    
    def create_face_dataset(self, source_video: str, target_video: str, 
                          output_dir: str, sample_rate: int = 1) -> Dict[str, Any]:
        """
        얼굴 교체를 위한 데이터셋 생성
        
        Args:
            source_video: 소스 비디오 (교체될 얼굴)
            target_video: 타겟 비디오 (교체할 얼굴)
            output_dir: 출력 디렉토리
            sample_rate: 샘플링 비율
            
        Returns:
            dataset_info: 데이터셋 정보
        """
        self.logger.info("얼굴 데이터셋 생성 시작")
        
        # 출력 디렉토리 구조 생성
        source_dir = Path(output_dir) / "source"
        target_dir = Path(output_dir) / "target"
        metadata_dir = Path(output_dir) / "metadata"
        
        ensure_dir(str(source_dir))
        ensure_dir(str(target_dir))
        ensure_dir(str(metadata_dir))
        
        # 소스 비디오 처리
        self.logger.info("소스 비디오 처리 중...")
        source_metadata = self.align_faces(source_video, str(source_dir), sample_rate)
        
        # 타겟 비디오 처리
        self.logger.info("타겟 비디오 처리 중...")
        target_metadata = self.align_faces(target_video, str(target_dir), sample_rate)
        
        # 데이터셋 정보 생성
        dataset_info = {
            'source_video': source_video,
            'target_video': target_video,
            'source_faces': len(source_metadata),
            'target_faces': len(target_metadata),
            'sample_rate': sample_rate,
            'created_at': get_timestamp(),
            'source_dir': str(source_dir),
            'target_dir': str(target_dir),
            'metadata_dir': str(metadata_dir)
        }
        
        # 메타데이터 저장
        metadata_path = metadata_dir / "dataset_info.json"
        save_metadata(dataset_info, str(metadata_path))
        
        self.logger.info(f"데이터셋 생성 완료: 소스 {len(source_metadata)}개, 타겟 {len(target_metadata)}개")
        return dataset_info
    
    def extract_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        이미지에서 얼굴 랜드마크 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            landmarks: 랜드마크 좌표 또는 None
        """
        return self.landmark_detector.detect_landmarks(image)
    
    def align_single_face(self, image: np.ndarray, landmarks: np.ndarray, 
                         output_size: int = 256) -> np.ndarray:
        """
        단일 얼굴 정렬
        
        Args:
            image: 입력 이미지
            landmarks: 랜드마크 좌표
            output_size: 출력 크기
            
        Returns:
            aligned_face: 정렬된 얼굴
        """
        return self.landmark_detector.align_face(image, landmarks, output_size)
    
    def create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        랜드마크로부터 얼굴 마스크 생성
        
        Args:
            landmarks: 랜드마크 좌표
            image_shape: 이미지 크기
            
        Returns:
            mask: 얼굴 마스크
        """
        from ..common.utils import create_mask_from_landmarks
        return create_mask_from_landmarks(landmarks, image_shape)
    
    def _calculate_face_quality(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """
        얼굴 품질 점수 계산
        
        Args:
            image: 이미지
            landmarks: 랜드마크 좌표
            
        Returns:
            quality_score: 품질 점수
        """
        from ..common.utils import calculate_face_quality_score
        return calculate_face_quality_score(image, landmarks)
    
    def filter_high_quality_faces(self, metadata: List[Dict], 
                                quality_threshold: float = 0.7) -> List[Dict]:
        """
        고품질 얼굴만 필터링
        
        Args:
            metadata: 얼굴 메타데이터 리스트
            quality_threshold: 품질 임계값
            
        Returns:
            filtered_metadata: 필터링된 메타데이터
        """
        filtered = [
            data for data in metadata 
            if data.get('quality_score', 0) >= quality_threshold
        ]
        
        self.logger.info(f"품질 필터링: {len(metadata)} -> {len(filtered)}개")
        return filtered
    
    def augment_face_data(self, face_images: List[np.ndarray], 
                         augmentation_config: Dict[str, Any]) -> List[np.ndarray]:
        """
        얼굴 데이터 증강
        
        Args:
            face_images: 얼굴 이미지 리스트
            augmentation_config: 증강 설정
            
        Returns:
            augmented_images: 증강된 이미지 리스트
        """
        augmented_images = []
        
        for image in face_images:
            augmented_images.append(image)  # 원본 추가
            
            # 회전
            if augmentation_config.get('rotation', False):
                for angle in [-10, -5, 5, 10]:
                    rotated = self._rotate_image(image, angle)
                    augmented_images.append(rotated)
            
            # 밝기 조정
            if augmentation_config.get('brightness', False):
                for factor in [0.8, 1.2]:
                    brightened = self._adjust_brightness(image, factor)
                    augmented_images.append(brightened)
            
            # 노이즈 추가
            if augmentation_config.get('noise', False):
                noisy = self._add_noise(image)
                augmented_images.append(noisy)
        
        return augmented_images
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """밝기 조정"""
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    def _add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """노이즈 추가"""
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

