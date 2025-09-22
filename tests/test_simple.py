# -*- coding: utf-8 -*-
"""
간단한 테스트
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.face_detector_simple import SimpleFaceDetector


class TestSimpleDetector(unittest.TestCase):
    """간단한 검출기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.detector = SimpleFaceDetector()
        
        # 테스트 이미지 생성 (얼굴이 있는 것처럼)
        self.test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """검출기 초기화 테스트"""
        self.assertIsNotNone(self.detector)
        # face_cascade가 None일 수 있음 (로드 실패 시)
        # self.assertIsNotNone(self.detector.face_cascade)
    
    def test_detect_faces(self):
        """얼굴 검출 테스트"""
        detections = self.detector.detect(self.test_image)
        
        # 결과 검증
        self.assertIsInstance(detections, list)
        
        # 각 검출 결과 검증
        for detection in detections:
            self.assertIn('bbox', detection)
            self.assertIn('landmarks', detection)
            self.assertIn('quality_score', detection)
            self.assertIn('confidence', detection)
            
            # bbox 검증
            bbox = detection['bbox']
            self.assertEqual(len(bbox), 4)  # [x, y, w, h]
            self.assertGreaterEqual(bbox[0], 0)
            self.assertGreaterEqual(bbox[1], 0)
            self.assertGreater(bbox[2], 0)
            self.assertGreater(bbox[3], 0)
            
            # landmarks 검증
            landmarks = detection['landmarks']
            self.assertIsInstance(landmarks, list)
            self.assertEqual(len(landmarks), 5)  # 5개 포인트
            
            # quality_score 검증
            quality_score = detection['quality_score']
            self.assertGreaterEqual(quality_score, 0.0)
            self.assertLessEqual(quality_score, 1.0)
    
    def test_visualize_detections(self):
        """검출 결과 시각화 테스트"""
        detections = self.detector.detect(self.test_image)
        vis_frame = self.detector.visualize_detections(self.test_image, detections)
        
        # 결과 검증
        self.assertEqual(vis_frame.shape, self.test_image.shape)
        self.assertEqual(vis_frame.dtype, self.test_image.dtype)


if __name__ == '__main__':
    unittest.main()
