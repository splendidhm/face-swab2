"""
이미지 향상 모듈
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import filters, exposure, restoration
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEnhancer:
    """이미지 향상 클래스"""
    
    def __init__(self, method: str = 'unsharp_mask'):
        """
        Args:
            method: 향상 방법
        """
        self.method = method
    
    def enhance(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        이미지 향상
        
        Args:
            image: 입력 이미지
            strength: 향상 강도
            
        Returns:
            enhanced_image: 향상된 이미지
        """
        if self.method == 'unsharp_mask':
            return self._unsharp_mask(image, strength)
        elif self.method == 'clahe':
            return self._clahe(image, strength)
        elif self.method == 'gaussian':
            return self._gaussian_enhancement(image, strength)
        elif self.method == 'bilateral':
            return self._bilateral_enhancement(image, strength)
        else:
            return image
    
    def _unsharp_mask(self, image: np.ndarray, strength: float) -> np.ndarray:
        """언샤프 마스크"""
        # 가우시안 블러
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # 언샤프 마스크
        mask = image.astype(np.float32) - blurred.astype(np.float32)
        enhanced = image.astype(np.float32) + strength * mask
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _clahe(self, image: np.ndarray, strength: float) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Lab 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Lab에서 BGR로 변환
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _gaussian_enhancement(self, image: np.ndarray, strength: float) -> np.ndarray:
        """가우시안 향상"""
        # 가우시안 필터
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        
        # 향상
        enhanced = image.astype(np.float32) + strength * (image.astype(np.float32) - gaussian.astype(np.float32))
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _bilateral_enhancement(self, image: np.ndarray, strength: float) -> np.ndarray:
        """양방향 필터 향상"""
        # 양방향 필터
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 향상
        enhanced = image.astype(np.float32) + strength * (image.astype(np.float32) - bilateral.astype(np.float32))
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)


class NoiseReducer:
    """노이즈 감소 클래스"""
    
    def __init__(self, method: str = 'bilateral'):
        """
        Args:
            method: 노이즈 감소 방법
        """
        self.method = method
    
    def reduce_noise(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        노이즈 감소
        
        Args:
            image: 입력 이미지
            strength: 감소 강도
            
        Returns:
            denoised_image: 노이즈 감소된 이미지
        """
        if self.method == 'bilateral':
            return self._bilateral_filter(image, strength)
        elif self.method == 'gaussian':
            return self._gaussian_filter(image, strength)
        elif self.method == 'median':
            return self._median_filter(image, strength)
        elif self.method == 'nlm':
            return self._non_local_means(image, strength)
        else:
            return image
    
    def _bilateral_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """양방향 필터"""
        d = int(9 * strength)
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _gaussian_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """가우시안 필터"""
        kernel_size = int(5 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _median_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """중간값 필터"""
        kernel_size = int(5 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.medianBlur(image, kernel_size)
    
    def _non_local_means(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Non-local Means 필터"""
        h = 10 * strength
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


class EdgeEnhancer:
    """엣지 향상 클래스"""
    
    def __init__(self, method: str = 'canny'):
        """
        Args:
            method: 엣지 향상 방법
        """
        self.method = method
    
    def enhance_edges(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        엣지 향상
        
        Args:
            image: 입력 이미지
            strength: 향상 강도
            
        Returns:
            enhanced_image: 엣지 향상된 이미지
        """
        if self.method == 'canny':
            return self._canny_enhancement(image, strength)
        elif self.method == 'sobel':
            return self._sobel_enhancement(image, strength)
        elif self.method == 'laplacian':
            return self._laplacian_enhancement(image, strength)
        else:
            return image
    
    def _canny_enhancement(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Canny 엣지 향상"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 엣지를 원본 이미지에 추가
        edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(image, 1.0, edges_3d, strength, 0)
        
        return enhanced
    
    def _sobel_enhancement(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Sobel 엣지 향상"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel 엣지 검출
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        # 엣지를 원본 이미지에 추가
        sobel_3d = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(image, 1.0, sobel_3d, strength, 0)
        
        return enhanced
    
    def _laplacian_enhancement(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Laplacian 엣지 향상"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Laplacian 엣지 검출
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 엣지를 원본 이미지에 추가
        laplacian_3d = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(image, 1.0, laplacian_3d, strength, 0)
        
        return enhanced


class SuperResolution:
    """슈퍼 해상도 클래스"""
    
    def __init__(self, method: str = 'bicubic'):
        """
        Args:
            method: 슈퍼 해상도 방법
        """
        self.method = method
    
    def upscale(self, image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        이미지 업스케일링
        
        Args:
            image: 입력 이미지
            scale_factor: 스케일 팩터
            
        Returns:
            upscaled_image: 업스케일된 이미지
        """
        if self.method == 'bicubic':
            return self._bicubic_upscale(image, scale_factor)
        elif self.method == 'lanczos':
            return self._lanczos_upscale(image, scale_factor)
        elif self.method == 'nearest':
            return self._nearest_upscale(image, scale_factor)
        else:
            return image
    
    def _bicubic_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Bicubic 업스케일링"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _lanczos_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Lanczos 업스케일링"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def _nearest_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Nearest 업스케일링"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


class ImageEnhancementPipeline:
    """이미지 향상 파이프라인"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: 향상 설정
        """
        self.config = config
        
        # 컴포넌트 초기화
        self.enhancer = ImageEnhancer(
            method=config.get('enhancement', {}).get('method', 'unsharp_mask')
        )
        
        self.noise_reducer = NoiseReducer(
            method=config.get('noise_reduction', {}).get('method', 'bilateral')
        )
        
        self.edge_enhancer = EdgeEnhancer(
            method=config.get('edge_enhancement', {}).get('method', 'canny')
        )
        
        self.super_resolution = SuperResolution(
            method=config.get('super_resolution', {}).get('method', 'bicubic')
        )
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 향상 파이프라인
        
        Args:
            image: 입력 이미지
            
        Returns:
            enhanced_image: 향상된 이미지
        """
        enhanced = image.copy()
        
        # 노이즈 감소
        if self.config.get('noise_reduction', {}).get('enabled', True):
            strength = self.config.get('noise_reduction', {}).get('strength', 1.0)
            enhanced = self.noise_reducer.reduce_noise(enhanced, strength)
        
        # 이미지 향상
        if self.config.get('enhancement', {}).get('enabled', True):
            strength = self.config.get('enhancement', {}).get('strength', 1.0)
            enhanced = self.enhancer.enhance(enhanced, strength)
        
        # 엣지 향상
        if self.config.get('edge_enhancement', {}).get('enabled', True):
            strength = self.config.get('edge_enhancement', {}).get('strength', 1.0)
            enhanced = self.edge_enhancer.enhance_edges(enhanced, strength)
        
        # 슈퍼 해상도
        if self.config.get('super_resolution', {}).get('enabled', False):
            scale_factor = self.config.get('super_resolution', {}).get('scale_factor', 2.0)
            enhanced = self.super_resolution.upscale(enhanced, scale_factor)
        
        return enhanced
    
    def enhance_video(self, input_video: str, output_video: str) -> None:
        """
        비디오 향상
        
        Args:
            input_video: 입력 비디오
            output_video: 출력 비디오
        """
        from ..common.video_io import VideoIO
        
        video_io = VideoIO()
        
        # 비디오 정보 가져오기
        video_info = video_io.get_video_info(input_video)
        
        # 출력 비디오 설정
        output_fps = video_info['fps']
        output_size = (video_info['width'], video_info['height'])
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, output_fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"출력 비디오를 생성할 수 없습니다: {output_video}")
        
        # 프레임별 처리
        frame_count = 0
        for frame, _ in video_io.read_video_generator(input_video, 1):
            # 향상
            enhanced_frame = self.enhance_image(frame)
            
            # 프레임 저장
            out.write(enhanced_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"처리된 프레임: {frame_count}")
        
        out.release()
        print(f"비디오 향상 완료: {output_video}")

