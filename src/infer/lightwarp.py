# -*- coding: utf-8 -*-
"""
LightWarp 모드 - 기하학적 변환 + 색상 전달
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from scipy.spatial import Delaunay
from skimage import exposure
import scipy.ndimage as ndi


class LightWarpProcessor:
    """LightWarp 처리 클래스"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def apply_target_to_frame(self, frame_bgr: np.ndarray, target_img: np.ndarray, 
                            dest_landmarks: np.ndarray, mode: str = 'lightwarp') -> np.ndarray:
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
        if mode != 'lightwarp':
            raise ValueError(f"지원하지 않는 모드: {mode}")
        
        # 타겟 이미지에서 랜드마크 추출
        target_landmarks = self._extract_target_landmarks(target_img)
        
        if target_landmarks is None:
            self.logger.warning("타겟 이미지에서 랜드마크를 추출할 수 없습니다")
            return frame_bgr
        
        # Delaunay 삼각분할 기반 변형
        warped_face = self.warp_delaunay(
            target_img, target_landmarks, dest_landmarks, frame_bgr.shape[:2]
        )
        
        # 색상 전달
        color_matched_face = self.color_transfer(
            warped_face, frame_bgr, dest_landmarks
        )
        
        # Poisson 블렌딩
        result_frame = self.paste_with_poisson(
            color_matched_face, frame_bgr, dest_landmarks
        )
        
        return result_frame
    
    def _extract_target_landmarks(self, target_img: np.ndarray) -> Optional[np.ndarray]:
        """
        타겟 이미지에서 랜드마크 추출
        
        Args:
            target_img: 타겟 이미지
            
        Returns:
            landmarks: 랜드마크 좌표 또는 None
        """
        # 간단한 얼굴 검출기 사용
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
        
        # 첫 번째 얼굴 사용
        x, y, w, h = faces[0]
        
        # 간단한 랜드마크 생성
        landmarks = [
            [x + w//4, y + h//3],      # 왼쪽 눈
            [x + 3*w//4, y + h//3],    # 오른쪽 눈
            [x + w//2, y + h//2],      # 코
            [x + w//3, y + 2*h//3],    # 왼쪽 입
            [x + 2*w//3, y + 2*h//3],  # 오른쪽 입
        ]
        
        return np.array(landmarks)
    
    def warp_delaunay(self, src_img: np.ndarray, src_points: np.ndarray, 
                     dst_points: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
        """
        Delaunay 삼각분할 기반 변형
        
        Args:
            src_img: 소스 이미지
            src_points: 소스 포인트
            dst_points: 대상 포인트
            out_shape: 출력 크기
            
        Returns:
            warped_img: 변형된 이미지
        """
        # 주요 랜드마크 포인트 선택 (눈, 코, 입)
        key_indices = self._get_key_landmark_indices()
        src_key_points = src_points[key_indices]
        dst_key_points = dst_points[key_indices]
        
        # 경계 포인트 추가
        src_boundary = self._get_boundary_points(src_img.shape[:2])
        dst_boundary = self._get_boundary_points(out_shape)
        
        # 모든 포인트 결합
        src_all_points = np.vstack([src_key_points, src_boundary])
        dst_all_points = np.vstack([dst_key_points, dst_boundary])
        
        # Delaunay 삼각분할
        tri = Delaunay(dst_all_points)
        
        # 변형된 이미지 초기화
        warped_img = np.zeros((out_shape[0], out_shape[1], 3), dtype=np.uint8)
        
        # 각 삼각형에 대해 변형 수행
        for simplex in tri.simplices:
            src_triangle = src_all_points[simplex]
            dst_triangle = dst_all_points[simplex]
            
            # 아핀 변환 행렬 계산
            transform_matrix = cv2.getAffineTransform(
                src_triangle[:3].astype(np.float32),
                dst_triangle[:3].astype(np.float32)
            )
            
            # 삼각형 마스크 생성
            mask = np.zeros(out_shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [dst_triangle.astype(np.int32)], 255)
            
            # 변형 적용
            warped_triangle = cv2.warpAffine(
                src_img, transform_matrix, (out_shape[1], out_shape[0])
            )
            
            # 마스크 적용하여 합성
            warped_img = np.where(mask[..., np.newaxis] > 0, warped_triangle, warped_img)
        
        return warped_img
    
    def _get_key_landmark_indices(self) -> List[int]:
        """주요 랜드마크 인덱스 반환"""
        # 간단한 5개 포인트 사용
        return [0, 1, 2, 3, 4]
    
    def _get_boundary_points(self, shape: Tuple[int, int]) -> np.ndarray:
        """경계 포인트 생성"""
        h, w = shape
        return np.array([
            [0, 0], [w//2, 0], [w-1, 0],
            [0, h//2], [w-1, h//2],
            [0, h-1], [w//2, h-1], [w-1, h-1]
        ])
    
    def color_transfer(self, src_region: np.ndarray, dst_frame: np.ndarray, 
                      dst_landmarks: np.ndarray) -> np.ndarray:
        """
        색상 전달
        
        Args:
            src_region: 소스 영역
            dst_frame: 대상 프레임
            dst_landmarks: 대상 랜드마크
            
        Returns:
            color_matched_region: 색상 전달된 영역
        """
        # 대상 영역 추출
        bbox = self._calculate_bbox(dst_landmarks)
        x, y, w, h = bbox
        
        if x < 0 or y < 0 or x + w > dst_frame.shape[1] or y + h > dst_frame.shape[0]:
            return src_region
        
        dst_region = dst_frame[y:y+h, x:x+w]
        
        if dst_region.size == 0:
            return src_region
        
        # 크기 맞추기
        if src_region.shape[:2] != dst_region.shape[:2]:
            src_region = cv2.resize(src_region, (dst_region.shape[1], dst_region.shape[0]))
        
        # 히스토그램 매칭
        color_matched = self._histogram_matching(src_region, dst_region)
        
        return color_matched
    
    def _calculate_bbox(self, landmarks: np.ndarray, margin: float = 0.1) -> List[int]:
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
    
    def _histogram_matching(self, src_img: np.ndarray, dst_img: np.ndarray) -> np.ndarray:
        """히스토그램 매칭"""
        result = src_img.copy()
        
        for c in range(3):
            src_channel = src_img[:, :, c]
            dst_channel = dst_img[:, :, c]
            
            # 히스토그램 매칭
            matched_channel = exposure.match_histograms(
                src_channel, dst_channel, channel_axis=None
            )
            
            result[:, :, c] = matched_channel
        
        return result
    
    def paste_with_poisson(self, src_face: np.ndarray, dst_frame: np.ndarray, 
                          dst_landmarks: np.ndarray, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Poisson 블렌딩으로 붙이기
        
        Args:
            src_face: 소스 얼굴
            dst_frame: 대상 프레임
            dst_landmarks: 대상 랜드마크
            center: 중심점
            
        Returns:
            blended_frame: 블렌딩된 프레임
        """
        # 마스크 생성
        mask = self._create_face_mask(dst_landmarks, dst_frame.shape[:2])
        
        # 중심점 계산
        if center is None:
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

