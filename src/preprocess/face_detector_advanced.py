# -*- coding: utf-8 -*-
"""
고급 얼굴 검출기 - MediaPipe 기반
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import logging


class FaceDetector:
    """MediaPipe 기반 얼굴 검출기"""
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: 최소 검출 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,  # 최대 10개 얼굴
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.logger = logging.getLogger(__name__)
    
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        프레임에서 얼굴 검출
        
        Args:
            frame_bgr: BGR 프레임
            
        Returns:
            detections: 검출된 얼굴 리스트
        """
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        results = self.face_mesh.process(frame_rgb)
        
        detections = []
        
        if results.multi_face_landmarks:
            h, w = frame_bgr.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 좌표 추출
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                # 바운딩 박스 계산
                bbox = self._calculate_bbox(landmarks)
                
                # 품질 점수 계산
                quality_score = self._calculate_quality_score(landmarks, bbox, w, h)
                
                detection = {
                    'bbox': bbox,
                    'landmarks': landmarks.tolist(),
                    'quality_score': quality_score,
                    'confidence': 1.0  # MediaPipe는 신뢰도 제공 안함
                }
                
                detections.append(detection)
        
        return detections
    
    def _calculate_bbox(self, landmarks: np.ndarray, margin: float = 0.1) -> List[int]:
        """
        랜드마크로부터 바운딩 박스 계산
        
        Args:
            landmarks: 랜드마크 좌표
            margin: 여백 비율
            
        Returns:
            bbox: [x, y, w, h]
        """
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
    
    def _calculate_quality_score(self, landmarks: np.ndarray, bbox: List[int], 
                               img_w: int, img_h: int) -> float:
        """
        얼굴 품질 점수 계산
        
        Args:
            landmarks: 랜드마크 좌표
            bbox: 바운딩 박스
            img_w: 이미지 너비
            img_h: 이미지 높이
            
        Returns:
            quality_score: 품질 점수 (0-1)
        """
        x, y, w, h = bbox
        
        # 크기 점수
        face_area = w * h
        img_area = img_w * img_h
        size_score = min(face_area / img_area * 4, 1.0)  # 최대 25% 면적
        
        # 대칭성 점수 (눈 좌표 기준)
        left_eye = landmarks[33]  # 왼쪽 눈
        right_eye = landmarks[362]  # 오른쪽 눈
        nose = landmarks[1]  # 코
        
        if len(landmarks) > 362:
            eye_distance = np.linalg.norm(right_eye - left_eye)
            left_dist = abs(left_eye[0] - nose[0])
            right_dist = abs(right_eye[0] - nose[0])
            symmetry_score = 1.0 - abs(left_dist - right_dist) / eye_distance
        else:
            symmetry_score = 0.5
        
        # 전체 품질 점수
        quality_score = (size_score + symmetry_score) / 2.0
        
        return max(0.0, min(1.0, quality_score))
    
    def get_face_landmarks_468(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        468개 랜드마크 추출
        
        Args:
            frame_bgr: BGR 프레임
            
        Returns:
            landmarks_list: 468개 랜드마크 리스트
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        landmarks_list = []
        
        if results.multi_face_landmarks:
            h, w = frame_bgr.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    landmarks.append([x, y])
                
                landmarks_list.append(np.array(landmarks))
        
        return landmarks_list
    
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
            landmarks = np.array(detection['landmarks'])
            quality_score = detection['quality_score']
            
            # 바운딩 박스 그리기
            x, y, w, h = bbox
            color = (0, 255, 0) if quality_score > 0.7 else (0, 255, 255)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # 품질 점수 텍스트
            cv2.putText(vis_frame, f"Q:{quality_score:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 랜드마크 그리기 (일부만)
            if len(landmarks) > 0:
                for point in landmarks[::10]:  # 10개마다 하나씩
                    cv2.circle(vis_frame, tuple(point.astype(int)), 2, (255, 0, 0), -1)
        
        return vis_frame

