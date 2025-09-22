"""
DeepFaceLab 모델 추론 클래스
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .model import create_model
from ..common.utils import setup_logging, ensure_dir, denormalize_image, normalize_image
from ..common.video_io import VideoIO


class DeepFaceLabInference:
    """DeepFaceLab 모델 추론 클래스"""
    
    def __init__(self, config: Dict, checkpoint_path: str):
        """
        Args:
            config: 추론 설정
            checkpoint_path: 체크포인트 파일 경로
        """
        self.config = config
        self.logger = setup_logging()
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = create_model(
            model_type=config['model']['name'],
            input_channels=3,
            latent_dim=config['model'].get('latent_dim', 512)
        ).to(self.device)
        
        # 체크포인트 로드
        self.load_checkpoint(checkpoint_path)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 비디오 IO 초기화
        self.video_io = VideoIO()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            tensor: 전처리된 텐서
        """
        # BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 크기 조정
        target_size = self.config['inference']['output_resolution']
        if image.shape[:2] != (target_size, target_size):
            image = cv2.resize(image, (target_size, target_size))
        
        # 정규화 (-1 to 1)
        image = (image.astype(np.float32) / 127.5) - 1.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image).to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        이미지 후처리
        
        Args:
            tensor: 모델 출력 텐서
            
        Returns:
            image: 후처리된 이미지 (BGR)
        """
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
    
    def swap_face_single(self, source_image: np.ndarray, 
                        target_image: np.ndarray) -> np.ndarray:
        """
        단일 이미지 얼굴 교체
        
        Args:
            source_image: 소스 이미지 (교체될 얼굴)
            target_image: 타겟 이미지 (교체할 얼굴)
            
        Returns:
            result_image: 결과 이미지
        """
        with torch.no_grad():
            # 소스 이미지 전처리
            source_tensor = self.preprocess_image(source_image)
            
            # 모델 추론
            if self.config['model']['name'] == 'SAE':
                output, _ = self.model(source_tensor)
            elif self.config['model']['name'] == 'LIAE':
                output, _, _ = self.model(source_tensor)
            elif self.config['model']['name'] == 'DF':
                output, _, _ = self.model(source_tensor)
            
            # 후처리
            swapped_face = self.postprocess_image(output)
            
            # 타겟 이미지에 블렌딩
            result_image = self._blend_faces(swapped_face, target_image)
            
            return result_image
    
    def swap_face_batch(self, source_images: List[np.ndarray], 
                       target_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        배치 이미지 얼굴 교체
        
        Args:
            source_images: 소스 이미지 리스트
            target_images: 타겟 이미지 리스트
            
        Returns:
            result_images: 결과 이미지 리스트
        """
        results = []
        
        for source_img, target_img in zip(source_images, target_images):
            result = self.swap_face_single(source_img, target_img)
            results.append(result)
        
        return results
    
    def swap_face_video(self, source_video: str, target_video: str, 
                       output_video: str, sample_rate: int = 1) -> None:
        """
        비디오 얼굴 교체
        
        Args:
            source_video: 소스 비디오 (교체될 얼굴)
            target_video: 타겟 비디오 (교체할 얼굴)
            output_video: 출력 비디오
            sample_rate: 샘플링 비율
        """
        self.logger.info(f"비디오 얼굴 교체 시작: {source_video} -> {target_video}")
        
        # 비디오 정보 가져오기
        source_info = self.video_io.get_video_info(source_video)
        target_info = self.video_io.get_video_info(target_video)
        
        # 출력 비디오 설정
        output_fps = target_info['fps']
        output_size = (target_info['width'], target_info['height'])
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, output_fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"출력 비디오를 생성할 수 없습니다: {output_video}")
        
        # 프레임별 처리
        source_frames = []
        target_frames = []
        
        # 소스 프레임 수집
        for frame, _ in self.video_io.read_video_generator(source_video, sample_rate):
            source_frames.append(frame)
        
        # 타겟 프레임 처리
        frame_count = 0
        for target_frame, frame_idx in self.video_io.read_video_generator(target_video, sample_rate):
            if frame_count < len(source_frames):
                source_frame = source_frames[frame_count]
                
                # 얼굴 교체
                result_frame = self.swap_face_single(source_frame, target_frame)
                
                # 프레임 저장
                out.write(result_frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    self.logger.info(f"처리된 프레임: {frame_count}")
        
        out.release()
        self.logger.info(f"비디오 얼굴 교체 완료: {output_video}")
    
    def _blend_faces(self, swapped_face: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        """
        교체된 얼굴을 타겟 이미지에 블렌딩
        
        Args:
            swapped_face: 교체된 얼굴
            target_image: 타겟 이미지
            
        Returns:
            blended_image: 블렌딩된 이미지
        """
        # 타겟 이미지 크기로 얼굴 크기 조정
        target_h, target_w = target_image.shape[:2]
        face_h, face_w = swapped_face.shape[:2]
        
        # 얼굴 크기 조정
        scale = min(target_w / face_w, target_h / face_h) * 0.8
        new_face_w = int(face_w * scale)
        new_face_h = int(face_h * scale)
        
        resized_face = cv2.resize(swapped_face, (new_face_w, new_face_h))
        
        # 타겟 이미지 중앙에 배치
        result_image = target_image.copy()
        
        start_y = (target_h - new_face_h) // 2
        start_x = (target_w - new_face_w) // 2
        end_y = start_y + new_face_h
        end_x = start_x + new_face_w
        
        # 마스크 생성 (타원형)
        mask = np.zeros((new_face_h, new_face_w), dtype=np.uint8)
        cv2.ellipse(mask, (new_face_w//2, new_face_h//2), 
                   (new_face_w//2, new_face_h//2), 0, 0, 360, 255, -1)
        
        # 가우시안 블러로 마스크 부드럽게
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask.astype(np.float32) / 255.0
        
        # 블렌딩
        for c in range(3):
            result_image[start_y:end_y, start_x:end_x, c] = (
                resized_face[:, :, c] * mask + 
                result_image[start_y:end_y, start_x:end_x, c] * (1 - mask)
            )
        
        return result_image
    
    def apply_color_transfer(self, source_image: np.ndarray, 
                           target_image: np.ndarray, 
                           method: str = 'rct') -> np.ndarray:
        """
        색상 전달 적용
        
        Args:
            source_image: 소스 이미지
            target_image: 타겟 이미지
            method: 색상 전달 방법
            
        Returns:
            transferred_image: 색상 전달된 이미지
        """
        if method == 'rct':
            return self._reinhard_color_transfer(source_image, target_image)
        elif method == 'lct':
            return self._linear_color_transfer(source_image, target_image)
        else:
            return target_image
    
    def _reinhard_color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Reinhard 색상 전달"""
        # Lab 색공간으로 변환
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 평균과 표준편차 계산
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # 색상 전달
        result_lab = (target_lab - target_mean) * (source_std / target_std) + source_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        # BGR로 변환
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _linear_color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """선형 색상 전달"""
        # RGB로 변환
        source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype(np.float32)
        target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 평균과 공분산 계산
        source_mean = np.mean(source_rgb, axis=(0, 1))
        target_mean = np.mean(target_rgb, axis=(0, 1))
        
        # 공분산 행렬 계산
        source_flat = source_rgb.reshape(-1, 3)
        target_flat = target_rgb.reshape(-1, 3)
        
        source_cov = np.cov(source_flat.T)
        target_cov = np.cov(target_flat.T)
        
        # 색상 전달
        result_rgb = (target_rgb - target_mean) @ np.linalg.inv(target_cov) @ source_cov + source_mean
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
        
        # BGR로 변환
        result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result

