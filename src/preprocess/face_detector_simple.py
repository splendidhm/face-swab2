# -*- coding: utf-8 -*-
"""
간단한 얼굴 검출기 - OpenCV 기반
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import logging


class SimpleFaceDetector:
    """OpenCV 기반 얼굴 검출기"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: 최소 검출 신뢰도
        """
        self.min_detection_confidence = min_detection_confidence
        self.logger = logging.getLogger(__name__)
        
        # OpenCV Haar Cascade 초기화
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier()
        
        if not self.face_cascade.load(cascade_path):
            self.logger.warning(f"Haar cascade 파일을 로드할 수 없습니다: {cascade_path}")
            # 대체 방법으로 더미 검출기 생성
            self.face_cascade = None
    
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        프레임에서 얼굴 검출
        
        Args:
            frame_bgr: BGR 프레임
            
        Returns:
            detections: 검출된 얼굴 리스트
        """
        detections = []
        
        # CascadeClassifier가 로드되지 않은 경우 더미 검출
        if self.face_cascade is None:
            # 테스트용 더미 얼굴 생성
            h, w = frame_bgr.shape[:2]
            dummy_face = [w//4, h//4, w//2, h//2]  # 중앙에 얼굴 크기 영역
            faces = [dummy_face]
        else:
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
        
        for (x, y, w, h) in faces:
            # 바운딩 박스
            bbox = [x, y, w, h]
            
            # 간단한 랜드마크 생성 (실제로는 더 정교한 방법 필요)
            landmarks = self._generate_simple_landmarks(bbox)
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(frame_bgr, bbox)
            
            detection = {
                'bbox': bbox,
                'landmarks': landmarks,
                'quality_score': quality_score,
                'confidence': 1.0
            }
            
            detections.append(detection)
        
        return detections
    
    def _generate_simple_landmarks(self, bbox: List[int]) -> List[List[int]]:
        """
        간단한 랜드마크 생성
        
        Args:
            bbox: 바운딩 박스 [x, y, w, h]
            
        Returns:
            landmarks: 랜드마크 좌표 리스트
        """
        x, y, w, h = bbox
        
        # 기본 얼굴 특징점들
        landmarks = [
            [x + w//4, y + h//3],      # 왼쪽 눈
            [x + 3*w//4, y + h//3],    # 오른쪽 눈
            [x + w//2, y + h//2],      # 코
            [x + w//3, y + 2*h//3],    # 왼쪽 입
            [x + 2*w//3, y + 2*h//3],  # 오른쪽 입
        ]
        
        return landmarks
    
    def _calculate_quality_score(self, frame: np.ndarray, bbox: List[int]) -> float:
        """
        얼굴 품질 점수 계산
        
        Args:
            frame: 프레임
            bbox: 바운딩 박스
            
        Returns:
            quality_score: 품질 점수 (0-1)
        """
        x, y, w, h = bbox
        
        # 크기 점수
        face_area = w * h
        img_area = frame.shape[0] * frame.shape[1]
        size_score = min(face_area / img_area * 4, 1.0)
        
        # 위치 점수 (중앙에 가까울수록 높은 점수)
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        position_score = 1.0 - (distance / max_distance)
        
        # 전체 품질 점수
        quality_score = (size_score + position_score) / 2.0
        
        return max(0.0, min(1.0, quality_score))
    
    def visualize_detections(self, frame_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        검출 결과 시각화
        
        Args:
            frame_bgr: 원본 프레임
            detections: 검출 결과
            
        Returns:
            vis_frame: 시각화된 프레임
        """
        vis_frame = frame_bgr.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            landmarks = detection['landmarks']
            quality_score = detection['quality_score']
            
            x, y, w, h = bbox
            
            # 바운딩 박스 그리기
            color = (0, 255, 0) if quality_score > 0.7 else (0, 255, 255)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # 품질 점수 텍스트
            cv2.putText(vis_frame, f"Q:{quality_score:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 랜드마크 그리기
            for point in landmarks:
                cv2.circle(vis_frame, tuple(point), 3, (255, 0, 0), -1)
        
        return vis_frame
