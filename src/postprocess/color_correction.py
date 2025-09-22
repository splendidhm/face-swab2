"""
색상 보정 모듈
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy import ndimage
from skimage import exposure, color
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrector:
    """색상 보정 클래스"""
    
    def __init__(self, method: str = 'histogram_matching'):
        """
        Args:
            method: 색상 보정 방법
        """
        self.method = method
    
    def correct_color(self, source_image: np.ndarray, target_image: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        색상 보정 수행
        
        Args:
            source_image: 소스 이미지
            target_image: 타겟 이미지
            mask: 마스크 (선택적)
            
        Returns:
            corrected_image: 색상 보정된 이미지
        """
        if self.method == 'histogram_matching':
            return self._histogram_matching(source_image, target_image, mask)
        elif self.method == 'reinhard':
            return self._reinhard_transfer(source_image, target_image, mask)
        elif self.method == 'linear':
            return self._linear_transfer(source_image, target_image, mask)
        elif self.method == 'lct':
            return self._lct_transfer(source_image, target_image, mask)
        else:
            return source_image
    
    def _histogram_matching(self, source: np.ndarray, target: np.ndarray, 
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """히스토그램 매칭"""
        if mask is not None:
            # 마스크가 있는 경우 마스크 영역만 처리
            source_masked = source * mask[:, :, np.newaxis]
            target_masked = target * mask[:, :, np.newaxis]
        else:
            source_masked = source
            target_masked = target
        
        # 각 채널별 히스토그램 매칭
        result = source.copy()
        
        for c in range(3):
            source_channel = source_masked[:, :, c]
            target_channel = target_masked[:, :, c]
            
            # 히스토그램 매칭
            matched_channel = exposure.match_histograms(
                source_channel, target_channel, channel_axis=None
            )
            
            if mask is not None:
                # 마스크 영역만 적용
                result[:, :, c] = np.where(
                    mask[:, :, c] > 0, matched_channel, source[:, :, c]
                )
            else:
                result[:, :, c] = matched_channel
        
        return result
    
    def _reinhard_transfer(self, source: np.ndarray, target: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Reinhard 색상 전달"""
        # Lab 색공간으로 변환
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        if mask is not None:
            # 마스크 영역만 처리
            mask_3d = np.stack([mask] * 3, axis=2)
            source_masked = source_lab * mask_3d
            target_masked = target_lab * mask_3d
        else:
            source_masked = source_lab
            target_masked = target_lab
        
        # 평균과 표준편차 계산
        source_mean = np.mean(source_masked, axis=(0, 1))
        source_std = np.std(source_masked, axis=(0, 1))
        target_mean = np.mean(target_masked, axis=(0, 1))
        target_std = np.std(target_masked, axis=(0, 1))
        
        # 색상 전달
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        # BGR로 변환
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _linear_transfer(self, source: np.ndarray, target: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        """선형 색상 전달"""
        if mask is not None:
            source_masked = source * mask[:, :, np.newaxis]
            target_masked = target * mask[:, :, np.newaxis]
        else:
            source_masked = source
            target_masked = target
        
        # 평균과 공분산 계산
        source_mean = np.mean(source_masked, axis=(0, 1))
        target_mean = np.mean(target_masked, axis=(0, 1))
        
        # 공분산 행렬 계산
        source_flat = source_masked.reshape(-1, 3)
        target_flat = target_masked.reshape(-1, 3)
        
        source_cov = np.cov(source_flat.T)
        target_cov = np.cov(target_flat.T)
        
        # 색상 전달
        source_centered = source.astype(np.float32) - source_mean
        result = source_centered @ np.linalg.inv(source_cov) @ target_cov + target_mean
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _lct_transfer(self, source: np.ndarray, target: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear Color Transfer (LCT)"""
        if mask is not None:
            source_masked = source * mask[:, :, np.newaxis]
            target_masked = target * mask[:, :, np.newaxis]
        else:
            source_masked = source
            target_masked = target
        
        # RGB to Lab
        source_lab = cv2.cvtColor(source_masked, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_masked, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 평균과 공분산 계산
        source_mean = np.mean(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        
        source_flat = source_lab.reshape(-1, 3)
        target_flat = target_lab.reshape(-1, 3)
        
        source_cov = np.cov(source_flat.T)
        target_cov = np.cov(target_flat.T)
        
        # LCT 변환
        source_centered = source_lab - source_mean
        result_lab = source_centered @ np.linalg.inv(source_cov) @ target_cov + target_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        # Lab to BGR
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result


class TemporalSmoother:
    """시간적 스무딩 클래스"""
    
    def __init__(self, window_size: int = 5, method: str = 'gaussian'):
        """
        Args:
            window_size: 윈도우 크기
            method: 스무딩 방법
        """
        self.window_size = window_size
        self.method = method
        self.frame_buffer = []
    
    def smooth_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 스무딩
        
        Args:
            frame: 입력 프레임
            
        Returns:
            smoothed_frame: 스무딩된 프레임
        """
        self.frame_buffer.append(frame.copy())
        
        # 버퍼 크기 제한
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) < 3:
            return frame
        
        if self.method == 'gaussian':
            return self._gaussian_smoothing()
        elif self.method == 'median':
            return self._median_smoothing()
        elif self.method == 'bilateral':
            return self._bilateral_smoothing()
        else:
            return frame
    
    def _gaussian_smoothing(self) -> np.ndarray:
        """가우시안 스무딩"""
        frames = np.array(self.frame_buffer)
        
        # 가중치 계산
        weights = np.exp(-0.5 * ((np.arange(len(frames)) - len(frames)//2) / (len(frames)/3))**2)
        weights = weights / weights.sum()
        
        # 가중 평균
        smoothed = np.average(frames, axis=0, weights=weights)
        
        return smoothed.astype(np.uint8)
    
    def _median_smoothing(self) -> np.ndarray:
        """중간값 스무딩"""
        frames = np.array(self.frame_buffer)
        return np.median(frames, axis=0).astype(np.uint8)
    
    def _bilateral_smoothing(self) -> np.ndarray:
        """양방향 스무딩"""
        current_frame = self.frame_buffer[-1]
        
        # 양방향 필터 적용
        smoothed = cv2.bilateralFilter(current_frame, 9, 75, 75)
        
        return smoothed
    
    def reset(self):
        """버퍼 리셋"""
        self.frame_buffer = []


class SeamlessBlender:
    """Seamless 블렌딩 클래스"""
    
    def __init__(self, method: str = 'poisson'):
        """
        Args:
            method: 블렌딩 방법
        """
        self.method = method
    
    def blend(self, source: np.ndarray, target: np.ndarray, 
              mask: np.ndarray, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Seamless 블렌딩 수행
        
        Args:
            source: 소스 이미지
            target: 타겟 이미지
            mask: 마스크
            center: 중심점 (선택적)
            
        Returns:
            blended_image: 블렌딩된 이미지
        """
        if center is None:
            center = (source.shape[1] // 2, source.shape[0] // 2)
        
        if self.method == 'poisson':
            return self._poisson_blending(source, target, mask, center)
        elif self.method == 'seamless':
            return self._seamless_cloning(source, target, mask, center)
        elif self.method == 'mixed':
            return self._mixed_cloning(source, target, mask, center)
        else:
            return target
    
    def _poisson_blending(self, source: np.ndarray, target: np.ndarray, 
                         mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Poisson 블렌딩"""
        return cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    
    def _seamless_cloning(self, source: np.ndarray, target: np.ndarray, 
                         mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Seamless 클로닝"""
        return cv2.seamlessClone(source, target, mask, center, cv2.MIXED_CLONE)
    
    def _mixed_cloning(self, source: np.ndarray, target: np.ndarray, 
                      mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Mixed 클로닝"""
        return cv2.seamlessClone(source, target, mask, center, cv2.MONOCHROME_TRANSFER)


class PostProcessor:
    """후처리 통합 클래스"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: 후처리 설정
        """
        self.config = config
        
        # 컴포넌트 초기화
        self.color_corrector = ColorCorrector(
            method=config.get('color_correction', {}).get('method', 'histogram_matching')
        )
        
        self.temporal_smoother = TemporalSmoother(
            window_size=config.get('temporal_smoothing', {}).get('window_size', 5),
            method=config.get('temporal_smoothing', {}).get('method', 'gaussian')
        )
        
        self.seamless_blender = SeamlessBlender(
            method=config.get('blending', {}).get('method', 'poisson')
        )
    
    def process_frame(self, source_frame: np.ndarray, target_frame: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        프레임 후처리
        
        Args:
            source_frame: 소스 프레임
            target_frame: 타겟 프레임
            mask: 마스크 (선택적)
            
        Returns:
            processed_frame: 후처리된 프레임
        """
        # 색상 보정
        if self.config.get('color_correction', {}).get('enabled', True):
            source_frame = self.color_corrector.correct_color(
                source_frame, target_frame, mask
            )
        
        # 시간적 스무딩
        if self.config.get('temporal_smoothing', {}).get('enabled', True):
            source_frame = self.temporal_smoother.smooth_frame(source_frame)
        
        # Seamless 블렌딩
        if self.config.get('blending', {}).get('enabled', True) and mask is not None:
            source_frame = self.seamless_blender.blend(
                source_frame, target_frame, mask
            )
        
        return source_frame
    
    def process_video(self, source_video: str, target_video: str, 
                     output_video: str, mask_video: Optional[str] = None) -> None:
        """
        비디오 후처리
        
        Args:
            source_video: 소스 비디오
            target_video: 타겟 비디오
            output_video: 출력 비디오
            mask_video: 마스크 비디오 (선택적)
        """
        from ..common.video_io import VideoIO
        
        video_io = VideoIO()
        
        # 비디오 정보 가져오기
        target_info = video_io.get_video_info(target_video)
        
        # 출력 비디오 설정
        output_fps = target_info['fps']
        output_size = (target_info['width'], target_info['height'])
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, output_fps, output_size)
        
        if not out.isOpened():
            raise ValueError(f"출력 비디오를 생성할 수 없습니다: {output_video}")
        
        # 프레임별 처리
        frame_count = 0
        for (source_frame, _), (target_frame, _) in zip(
            video_io.read_video_generator(source_video, 1),
            video_io.read_video_generator(target_video, 1)
        ):
            # 마스크 로드 (있는 경우)
            mask = None
            if mask_video:
                # 마스크 비디오에서 프레임 로드
                pass  # 구현 필요
            
            # 후처리
            processed_frame = self.process_frame(source_frame, target_frame, mask)
            
            # 프레임 저장
            out.write(processed_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"처리된 프레임: {frame_count}")
        
        out.release()
        print(f"비디오 후처리 완료: {output_video}")
    
    def reset(self):
        """상태 리셋"""
        self.temporal_smoother.reset()

