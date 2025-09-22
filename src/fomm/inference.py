"""
FOMM 모델 추론 클래스
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .model import create_fomm_model
from ..common.utils import setup_logging, ensure_dir
from ..common.video_io import VideoIO


class FOMMInference:
    """FOMM 모델 추론 클래스"""
    
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
        self.model = create_fomm_model(
            num_kp=config.get('num_kp', 10),
            num_channels=3
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
    
    def animate_single(self, source_image: np.ndarray, 
                      driving_image: np.ndarray) -> np.ndarray:
        """
        단일 이미지 애니메이션
        
        Args:
            source_image: 소스 이미지 (애니메이션할 얼굴)
            driving_image: 드라이빙 이미지 (모션 소스)
            
        Returns:
            animated_image: 애니메이션된 이미지
        """
        with torch.no_grad():
            # 이미지 전처리
            source_tensor = self.preprocess_image(source_image)
            driving_tensor = self.preprocess_image(driving_image)
            
            # 모델 추론
            generated_tensor, kp_source, kp_driving, jacobian = self.model(
                source_tensor, driving_tensor
            )
            
            # 후처리
            animated_image = self.postprocess_image(generated_tensor)
            
            return animated_image
    
    def animate_sequence(self, source_image: np.ndarray, 
                        driving_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """
        시퀀스 애니메이션
        
        Args:
            source_image: 소스 이미지
            driving_sequence: 드라이빙 시퀀스
            
        Returns:
            animated_sequence: 애니메이션된 시퀀스
        """
        animated_sequence = []
        
        for driving_image in driving_sequence:
            animated_image = self.animate_single(source_image, driving_image)
            animated_sequence.append(animated_image)
        
        return animated_sequence
    
    def animate_video(self, source_image: np.ndarray, driving_video: str, 
                    output_video: str, sample_rate: int = 1) -> None:
        """
        비디오 애니메이션
        
        Args:
            source_image: 소스 이미지
            driving_video: 드라이빙 비디오
            output_video: 출력 비디오
            sample_rate: 샘플링 비율
        """
        self.logger.info(f"비디오 애니메이션 시작: {driving_video}")
        
        # 비디오 정보 가져오기
        driving_info = self.video_io.get_video_info(driving_video)
        
        # 출력 비디오 설정
        output_fps = driving_info['fps']
        output_size = (driving_info['width'], driving_info['height'])
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, output_fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"출력 비디오를 생성할 수 없습니다: {output_video}")
        
        # 프레임별 처리
        frame_count = 0
        for driving_frame, frame_idx in self.video_io.read_video_generator(driving_video, sample_rate):
            # 애니메이션
            animated_frame = self.animate_single(source_image, driving_frame)
            
            # 프레임 저장
            out.write(animated_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                self.logger.info(f"처리된 프레임: {frame_count}")
        
        out.release()
        self.logger.info(f"비디오 애니메이션 완료: {output_video}")
    
    def extract_keypoints(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 키포인트 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            keypoints: 키포인트 좌표
            jacobian: 자코비안 행렬
        """
        with torch.no_grad():
            # 이미지 전처리
            image_tensor = self.preprocess_image(image)
            
            # 키포인트 추출
            kp, jacobian = self.model.encode(image_tensor)
            
            # CPU로 이동 및 numpy 변환
            kp = kp.detach().cpu().numpy()[0]
            jacobian = jacobian.detach().cpu().numpy()[0]
            
            return kp, jacobian
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: np.ndarray, 
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        키포인트 시각화
        
        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표
            color: 색상 (BGR)
            
        Returns:
            vis_image: 시각화된 이미지
        """
        vis_image = image.copy()
        
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(vis_image, (x, y), 3, color, -1)
        
        return vis_image
    
    def create_animation_gif(self, source_image: np.ndarray, driving_video: str, 
                            output_gif: str, duration: float = 2.0, 
                            sample_rate: int = 1) -> None:
        """
        GIF 애니메이션 생성
        
        Args:
            source_image: 소스 이미지
            driving_video: 드라이빙 비디오
            output_gif: 출력 GIF 파일
            duration: 애니메이션 지속 시간 (초)
            sample_rate: 샘플링 비율
        """
        import imageio
        
        self.logger.info(f"GIF 애니메이션 생성 시작: {driving_video}")
        
        # 프레임 수집
        frames = []
        frame_count = 0
        
        for driving_frame, frame_idx in self.video_io.read_video_generator(driving_video, sample_rate):
            # 애니메이션
            animated_frame = self.animate_single(source_image, driving_frame)
            
            # RGB로 변환 (GIF용)
            animated_frame_rgb = cv2.cvtColor(animated_frame, cv2.COLOR_BGR2RGB)
            frames.append(animated_frame_rgb)
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                self.logger.info(f"처리된 프레임: {frame_count}")
        
        # GIF 생성
        if frames:
            fps = len(frames) / duration
            imageio.mimsave(output_gif, frames, fps=fps)
            self.logger.info(f"GIF 애니메이션 완료: {output_gif}")
    
    def batch_animate(self, source_images: List[np.ndarray], 
                     driving_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        배치 애니메이션
        
        Args:
            source_images: 소스 이미지 리스트
            driving_images: 드라이빙 이미지 리스트
            
        Returns:
            animated_images: 애니메이션된 이미지 리스트
        """
        results = []
        
        for source_img, driving_img in zip(source_images, driving_images):
            animated_img = self.animate_single(source_img, driving_img)
            results.append(animated_img)
        
        return results

