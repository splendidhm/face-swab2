# -*- coding: utf-8 -*-
"""
얼굴 트래킹 매니저
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
import time


class FaceTrackerManager:
    """얼굴 트래킹 매니저"""
    
    def __init__(self, tracker_type: str = 'CSRT', 
                 lost_face_timeout: int = 30,
                 iou_threshold: float = 0.3):
        """
        Args:
            tracker_type: 트래커 타입 ('CSRT', 'KCF', 'MOSSE')
            lost_face_timeout: 얼굴 손실 타임아웃 (프레임 수)
            iou_threshold: IoU 임계값
        """
        self.tracker_type = tracker_type
        self.lost_face_timeout = lost_face_timeout
        self.iou_threshold = iou_threshold
        
        # 트래커 딕셔너리 {face_id: tracker}
        self.trackers = {}
        
        # 얼굴 메타데이터 {face_id: metadata}
        self.face_metadata = {}
        
        # 다음 사용 가능한 face_id
        self.next_face_id = 0
        
        # 프레임 카운터
        self.frame_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _create_tracker(self):
        """새 트래커 생성"""
        try:
            if self.tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create()
            elif self.tracker_type == 'KCF':
                return cv2.TrackerKCF_create()
            elif self.tracker_type == 'MOSSE':
                return cv2.TrackerMOSSE_create()
            else:
                raise ValueError(f"지원하지 않는 트래커 타입: {self.tracker_type}")
        except AttributeError:
            # OpenCV 버전에 따라 다른 API 사용
            try:
                if self.tracker_type == 'CSRT':
                    return cv2.legacy.TrackerCSRT_create()
                elif self.tracker_type == 'KCF':
                    return cv2.legacy.TrackerKCF_create()
                elif self.tracker_type == 'MOSSE':
                    return cv2.legacy.TrackerMOSSE_create()
                else:
                    raise ValueError(f"지원하지 않는 트래커 타입: {self.tracker_type}")
            except AttributeError:
                # 최신 OpenCV 버전
                if self.tracker_type == 'CSRT':
                    return cv2.TrackerCSRT.create()
                elif self.tracker_type == 'KCF':
                    return cv2.TrackerKCF.create()
                elif self.tracker_type == 'MOSSE':
                    return cv2.TrackerMOSSE.create()
                else:
                    raise ValueError(f"지원하지 않는 트래커 타입: {self.tracker_type}")
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        두 바운딩 박스의 IoU 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스 [x, y, w, h]
            bbox2: 두 번째 바운딩 박스 [x, y, w, h]
            
        Returns:
            iou: IoU 값
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 교집합 계산
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _find_best_match(self, detections: List[Dict], 
                        existing_faces: Dict[int, Dict]) -> Dict[int, int]:
        """
        검출된 얼굴과 기존 얼굴 매칭
        
        Args:
            detections: 검출된 얼굴 리스트
            existing_faces: 기존 얼굴 딕셔너리
            
        Returns:
            matches: {detection_idx: face_id} 매칭 결과
        """
        matches = {}
        used_face_ids = set()
        
        # IoU 기반 매칭
        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_face_id = None
            
            for face_id, face_data in existing_faces.items():
                if face_id in used_face_ids:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], face_data['bbox'])
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_face_id = face_id
            
            if best_face_id is not None:
                matches[det_idx] = best_face_id
                used_face_ids.add(best_face_id)
        
        return matches
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        트래커 업데이트
        
        Args:
            detections: 검출된 얼굴 리스트
            frame: 현재 프레임
            
        Returns:
            tracked_faces: 트래킹된 얼굴 리스트
        """
        self.frame_count += 1
        tracked_faces = []
        
        # 기존 얼굴 메타데이터 업데이트
        existing_faces = {}
        for face_id, metadata in self.face_metadata.items():
            if metadata['last_seen'] + self.lost_face_timeout >= self.frame_count:
                existing_faces[face_id] = metadata
        
        # 검출된 얼굴과 기존 얼굴 매칭
        matches = self._find_best_match(detections, existing_faces)
        
        # 매칭된 얼굴 업데이트
        for det_idx, face_id in matches.items():
            detection = detections[det_idx]
            
            # 트래커 업데이트
            if face_id in self.trackers:
                success, bbox = self.trackers[face_id].update(frame)
                if success:
                    # 트래킹 성공
                    tracked_face = {
                        'face_id': face_id,
                        'bbox': bbox,
                        'landmarks': detection['landmarks'],
                        'quality_score': detection['quality_score'],
                        'confidence': 1.0,
                        'tracking_status': 'tracked'
                    }
                    tracked_faces.append(tracked_face)
                    
                    # 메타데이터 업데이트
                    self.face_metadata[face_id].update({
                        'bbox': bbox,
                        'landmarks': detection['landmarks'],
                        'last_seen': self.frame_count,
                        'frame_count': self.face_metadata[face_id]['frame_count'] + 1
                    })
                else:
                    # 트래킹 실패 - 새 트래커 초기화
                    self._initialize_tracker(face_id, detection, frame)
            else:
                # 새 트래커 초기화
                self._initialize_tracker(face_id, detection, frame)
        
        # 매칭되지 않은 검출에 대해 새 얼굴 생성
        for det_idx, detection in enumerate(detections):
            if det_idx not in matches:
                face_id = self._create_new_face(detection, frame)
                tracked_face = {
                    'face_id': face_id,
                    'bbox': detection['bbox'],
                    'landmarks': detection['landmarks'],
                    'quality_score': detection['quality_score'],
                    'confidence': 1.0,
                    'tracking_status': 'new'
                }
                tracked_faces.append(tracked_face)
        
        # 사용되지 않은 트래커 정리
        self._cleanup_unused_trackers()
        
        return tracked_faces
    
    def _initialize_tracker(self, face_id: int, detection: Dict, frame: np.ndarray):
        """트래커 초기화"""
        bbox = detection['bbox']
        x, y, w, h = bbox
        
        # OpenCV 트래커는 (x, y, w, h) 형식
        tracker = self._create_tracker()
        tracker.init(frame, (x, y, w, h))
        
        self.trackers[face_id] = tracker
    
    def _create_new_face(self, detection: Dict, frame: np.ndarray) -> int:
        """새 얼굴 생성"""
        face_id = self.next_face_id
        self.next_face_id += 1
        
        # 트래커 초기화
        self._initialize_tracker(face_id, detection, frame)
        
        # 메타데이터 생성
        self.face_metadata[face_id] = {
            'bbox': detection['bbox'],
            'landmarks': detection['landmarks'],
            'first_seen': self.frame_count,
            'last_seen': self.frame_count,
            'frame_count': 1,
            'average_bbox': detection['bbox'].copy()
        }
        
        return face_id
    
    def _cleanup_unused_trackers(self):
        """사용되지 않은 트래커 정리"""
        current_time = self.frame_count
        to_remove = []
        
        for face_id, metadata in self.face_metadata.items():
            if current_time - metadata['last_seen'] > self.lost_face_timeout:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            if face_id in self.trackers:
                del self.trackers[face_id]
            if face_id in self.face_metadata:
                del self.face_metadata[face_id]
    
    def get_face_tracks(self) -> Dict[int, Dict]:
        """
        얼굴 트랙 정보 반환
        
        Returns:
            tracks: 얼굴 트랙 딕셔너리
        """
        tracks = {}
        
        for face_id, metadata in self.face_metadata.items():
            tracks[face_id] = {
                'frames': list(range(metadata['first_seen'], metadata['last_seen'] + 1)),
                'meta': {
                    'average_bbox': metadata['average_bbox'],
                    'frame_count': metadata['frame_count'],
                    'first_seen': metadata['first_seen'],
                    'last_seen': metadata['last_seen']
                }
            }
        
        return tracks
    
    def reset(self):
        """트래커 리셋"""
        self.trackers.clear()
        self.face_metadata.clear()
        self.next_face_id = 0
        self.frame_count = 0

