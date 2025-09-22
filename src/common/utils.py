"""
공통 유틸리티 함수들
"""
import os
import yaml
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨
        log_file: 로그 파일 경로
        
    Returns:
        logger: 설정된 로거
    """
    logger = logging.getLogger('face_swap')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        config: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    YAML 설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 저장 경로
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def ensure_dir(path: str) -> None:
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        path: 디렉토리 경로
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """
    현재 시간을 문자열로 반환
    
    Returns:
        timestamp: YYYYMMDD_HHMMSS 형식의 시간 문자열
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """
    메타데이터를 JSON 파일로 저장
    
    Args:
        metadata: 메타데이터 딕셔너리
        output_path: 저장 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    JSON 메타데이터 파일 로드
    
    Args:
        metadata_path: 메타데이터 파일 경로
        
    Returns:
        metadata: 메타데이터 딕셔너리
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect: bool = True) -> np.ndarray:
    """
    이미지 크기 조정
    
    Args:
        image: 입력 이미지
        target_size: 목표 크기 (width, height)
        keep_aspect: 종횡비 유지 여부
        
    Returns:
        resized_image: 크기 조정된 이미지
    """
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 종횡비 계산
        aspect_ratio = w / h
        target_aspect = target_w / target_h
        
        if aspect_ratio > target_aspect:
            # 너비가 더 큰 경우
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # 높이가 더 큰 경우
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # 이미지 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 패딩 추가
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


def normalize_image(image: np.ndarray, mean: List[float] = [0.5, 0.5, 0.5], 
                     std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
    """
    이미지 정규화
    
    Args:
        image: 입력 이미지 (0-255)
        mean: 평균값
        std: 표준편차
        
    Returns:
        normalized_image: 정규화된 이미지 (-1~1)
    """
    # 0-1 범위로 변환
    normalized = image.astype(np.float32) / 255.0
    
    # 정규화
    for i in range(3):
        normalized[:, :, i] = (normalized[:, :, i] - mean[i]) / std[i]
    
    return normalized


def denormalize_image(image: np.ndarray, mean: List[float] = [0.5, 0.5, 0.5], 
                     std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
    """
    정규화된 이미지를 원래 범위로 복원
    
    Args:
        image: 정규화된 이미지 (-1~1)
        mean: 평균값
        std: 표준편차
        
    Returns:
        denormalized_image: 복원된 이미지 (0-255)
    """
    # 정규화 역변환
    denormalized = image.copy()
    for i in range(3):
        denormalized[:, :, i] = denormalized[:, :, i] * std[i] + mean[i]
    
    # 0-255 범위로 변환
    denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
    
    return denormalized


def create_mask_from_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int], 
                              dilate_kernel_size: int = 5) -> np.ndarray:
    """
    랜드마크로부터 마스크 생성
    
    Args:
        landmarks: 랜드마크 좌표
        image_shape: 이미지 크기 (height, width)
        dilate_kernel_size: 팽창 커널 크기
        
    Returns:
        mask: 이진 마스크
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 랜드마크로부터 컨벡스 헐 계산
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillPoly(mask, [hull], 255)
    
    # 팽창 연산으로 마스크 확장
    if dilate_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (dilate_kernel_size, dilate_kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def blend_images(source: np.ndarray, target: np.ndarray, mask: np.ndarray, 
                method: str = 'poisson') -> np.ndarray:
    """
    두 이미지를 블렌딩
    
    Args:
        source: 소스 이미지
        target: 타겟 이미지
        mask: 블렌딩 마스크
        method: 블렌딩 방법 ('poisson', 'seamless', 'linear')
        
    Returns:
        blended_image: 블렌딩된 이미지
    """
    if method == 'poisson':
        # Poisson 블렌딩
        center = (source.shape[1] // 2, source.shape[0] // 2)
        return cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    
    elif method == 'seamless':
        # Seamless 블렌딩
        center = (source.shape[1] // 2, source.shape[0] // 2)
        return cv2.seamlessClone(source, target, mask, center, cv2.MIXED_CLONE)
    
    elif method == 'linear':
        # 선형 블렌딩
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = np.stack([mask_norm] * 3, axis=2)
        
        blended = source.astype(np.float32) * mask_norm + target.astype(np.float32) * (1 - mask_norm)
        return blended.astype(np.uint8)
    
    else:
        raise ValueError(f"지원하지 않는 블렌딩 방법: {method}")


def calculate_face_quality_score(image: np.ndarray, landmarks: np.ndarray) -> float:
    """
    얼굴 품질 점수 계산
    
    Args:
        image: 얼굴 이미지
        landmarks: 랜드마크 좌표
        
    Returns:
        quality_score: 품질 점수 (0-1)
    """
    # 얼굴 크기 점수
    bbox = cv2.boundingRect(landmarks.astype(np.int32))
    face_area = bbox[2] * bbox[3]
    image_area = image.shape[0] * image.shape[1]
    size_score = min(face_area / image_area, 1.0)
    
    # 대칭성 점수 (간단한 구현)
    left_eye = landmarks[0] if len(landmarks) > 0 else None
    right_eye = landmarks[1] if len(landmarks) > 1 else None
    
    if left_eye is not None and right_eye is not None:
        eye_distance = np.linalg.norm(right_eye - left_eye)
        image_center_x = image.shape[1] / 2
        left_eye_dist = abs(left_eye[0] - image_center_x)
        right_eye_dist = abs(right_eye[0] - image_center_x)
        symmetry_score = 1.0 - abs(left_eye_dist - right_eye_dist) / eye_distance
    else:
        symmetry_score = 0.5
    
    # 전체 품질 점수
    quality_score = (size_score + symmetry_score) / 2.0
    
    return max(0.0, min(1.0, quality_score))

