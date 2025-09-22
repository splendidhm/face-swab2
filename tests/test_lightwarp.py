# -*- coding: utf-8 -*-
"""
LightWarp 모듈 단위 테스트
"""
import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infer.lightwarp import LightWarpProcessor


class TestLightWarp(unittest.TestCase):
    """LightWarp 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.processor = LightWarpProcessor()
        
        # 테스트 이미지 생성
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_landmarks = np.array([
            [20, 20], [80, 20], [50, 50], [20, 80], [80, 80]
        ])
    
    def test_warp_delaunay(self):
        """Delaunay 변형 테스트"""
        src_points = self.test_landmarks
        dst_points = src_points + np.array([10, 10])  # 약간 이동
        
        warped = self.processor.warp_delaunay(
            self.test_image, src_points, dst_points, (100, 100)
        )
        
        # 결과 검증
        self.assertEqual(warped.shape, (100, 100, 3))
        self.assertEqual(warped.dtype, np.uint8)
    
    def test_color_transfer(self):
        """색상 전달 테스트"""
        src_region = self.test_image
        dst_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        dst_landmarks = self.test_landmarks * 2  # 더 큰 영역
        
        color_matched = self.processor.color_transfer(
            src_region, dst_frame, dst_landmarks
        )
        
        # 결과 검증
        self.assertEqual(color_matched.shape, src_region.shape)
        self.assertEqual(color_matched.dtype, np.uint8)
    
    def test_calculate_bbox(self):
        """바운딩 박스 계산 테스트"""
        bbox = self.processor._calculate_bbox(self.test_landmarks)
        
        # 결과 검증
        self.assertEqual(len(bbox), 4)  # [x, y, w, h]
        self.assertGreaterEqual(bbox[0], 0)  # x >= 0
        self.assertGreaterEqual(bbox[1], 0)  # y >= 0
        self.assertGreater(bbox[2], 0)      # w > 0
        self.assertGreater(bbox[3], 0)      # h > 0
    
    def test_create_face_mask(self):
        """얼굴 마스크 생성 테스트"""
        mask = self.processor._create_face_mask(
            self.test_landmarks, (100, 100)
        )
        
        # 결과 검증
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.any(mask > 0))  # 마스크가 비어있지 않음
    
    def test_calculate_center(self):
        """중심점 계산 테스트"""
        center = self.processor._calculate_center(self.test_landmarks)
        
        # 결과 검증
        self.assertEqual(len(center), 2)  # (x, y)
        self.assertIsInstance(center[0], int)
        self.assertIsInstance(center[1], int)
    
    def test_apply_target_to_frame(self):
        """타겟 적용 테스트"""
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        landmarks = self.test_landmarks * 2
        
        result = self.processor.apply_target_to_frame(
            frame, target, landmarks, 'lightwarp'
        )
        
        # 결과 검증
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)


if __name__ == '__main__':
    unittest.main()

