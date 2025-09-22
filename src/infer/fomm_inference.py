# -*- coding: utf-8 -*-
"""
FOMM 추론 모듈
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple
import logging
from pathlib import Path


class FOMMModel:
    """FOMM 모델 래퍼"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 디바이스 ('auto', 'cpu', 'cuda')
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._get_device(device)
        self.logger = logging.getLogger(__name__)
        
        # 모델 로드
        self.model = self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model(self):
        """모델 로드"""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"체크포인트 파일을 찾을 수 없습니다: {self.checkpoint_path}\n"
                f"FOMM 체크포인트를 다음 위치에 배치하세요:\n"
                f"  - {self.checkpoint_path}\n"
                f"체크포인트 다운로드 링크: https://github.com/AliaksandrSiarohin/first-order-model"
            )
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 모델 초기화 (실제 FOMM 모델 구조에 맞게 수정 필요)
            from ..fomm.model import create_fomm_model
            model = create_fomm_model()
            
            # 가중치 로드
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"FOMM 모델 로드 완료: {self.checkpoint_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}")
    
    def generate(self, source_img: np.ndarray, driving_crop: np.ndarray) -> np.ndarray:
        """
        FOMM 생성
        
        Args:
            source_img: 소스 이미지 (타겟 얼굴)
            driving_crop: 드라이빙 크롭 (원본 얼굴)
            
        Returns:
            generated_face: 생성된 얼굴
        """
        with torch.no_grad():
            # 이미지 전처리
            source_tensor = self._preprocess_image(source_img)
            driving_tensor = self._preprocess_image(driving_crop)
            
            # 모델 추론
            generated_tensor = self.model(source_tensor, driving_tensor)
            
            # 후처리
            generated_face = self._postprocess_image(generated_tensor)
            
            return generated_face
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 크기 조정
        image = cv2.resize(image, (256, 256))
        
        # 정규화 (-1 to 1)
        image = (image.astype(np.float32) / 127.5) - 1.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image).to(self.device)
    
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """이미지 후처리"""
        # CPU로 이동 및 numpy 변환
        image = tensor.detach().cpu().numpy()
        
        # 배치 차원 제거
        image = image[0]
        
        # CHW to HWC
        image = np.transpose(image, (1, 2, 0))
        
        # 정규화 역변환 (-1 to 1 -> 0 to 255)
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image


class FOMMProcessor:
    """FOMM 처리 클래스"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 디바이스
        """
        self.model = FOMMModel(checkpoint_path, device)
        self.logger = logging.getLogger(__name__)
    
    def apply_target_to_frame(self, frame_bgr: np.ndarray, target_img: np.ndarray, 
                            dest_landmarks: np.ndarray, mode: str = 'fomm') -> np.ndarray:
        """
        타겟 이미지를 프레임에 적용
        
        Args:
            frame_bgr: 대상 프레임
            target_img: 타겟 얼굴 이미지
            dest_landmarks: 대상 랜드마크
            mode: 처리 모드
            
        Returns:
            result_frame: 결과 프레임
        """
        if mode != 'fomm':
            raise ValueError(f"지원하지 않는 모드: {mode}")
        
        # 대상 영역 추출
        bbox = self._calculate_bbox(dest_landmarks)
        x, y, w, h = bbox
        
        if x < 0 or y < 0 or x + w > frame_bgr.shape[1] or y + h > frame_bgr.shape[0]:
            return frame_bgr
        
        driving_crop = frame_bgr[y:y+h, x:x+w]
        
        if driving_crop.size == 0:
            return frame_bgr
        
        # FOMM 생성
        generated_face = self.model.generate(target_img, driving_crop)
        
        # 크기 맞추기
        if generated_face.shape[:2] != driving_crop.shape[:2]:
            generated_face = cv2.resize(generated_face, (driving_crop.shape[1], driving_crop.shape[0]))
        
        # Poisson 블렌딩
        result_frame = self._paste_with_poisson(
            generated_face, frame_bgr, dest_landmarks
        )
        
        return result_frame
    
    def _calculate_bbox(self, landmarks: np.ndarray, margin: float = 0.1) -> list:
        """바운딩 박스 계산"""
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        margin_x = width * margin
        margin_y = height * margin
        
        x = max(0, int(x_min - margin_x))
        y = max(0, int(y_min - margin_y))
        w = int(x_max + margin_x) - x
        h = int(y_max + margin_y) - y
        
        return [x, y, w, h]
    
    def _paste_with_poisson(self, src_face: np.ndarray, dst_frame: np.ndarray, 
                           dst_landmarks: np.ndarray) -> np.ndarray:
        """Poisson 블렌딩으로 붙이기"""
        # 마스크 생성
        mask = self._create_face_mask(dst_landmarks, dst_frame.shape[:2])
        
        # 중심점 계산
        center = self._calculate_center(dst_landmarks)
        
        # Poisson 블렌딩
        result = cv2.seamlessClone(
            src_face, dst_frame, mask, center, cv2.NORMAL_CLONE
        )
        
        return result
    
    def _create_face_mask(self, landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """얼굴 마스크 생성"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        # 컨벡스 헐 계산
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _calculate_center(self, landmarks: np.ndarray) -> Tuple[int, int]:
        """중심점 계산"""
        center = landmarks.mean(axis=0).astype(int)
        return tuple(center)

