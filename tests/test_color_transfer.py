# -*- coding: utf-8 -*-
"""
색상 전달 모듈 단위 테스트
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


class TestColorTransfer(unittest.TestCase):
    """색상 전달 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.processor = LightWarpProcessor()
        
        # 테스트 이미지 생성
        self.src_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.dst_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_histogram_matching(self):
        """히스토그램 매칭 테스트"""
        matched = self.processor._histogram_matching(self.src_image, self.dst_image)
        
        # 결과 검증
        self.assertEqual(matched.shape, self.src_image.shape)
        self.assertEqual(matched.dtype, self.src_image.dtype)
        
        # 값 범위 검증
        self.assertTrue(np.all(matched >= 0))
        self.assertTrue(np.all(matched <= 255))
    
    def test_color_transfer_basic(self):
        """기본 색상 전달 테스트"""
        result = self.processor.color_transfer(
            self.src_image, self.dst_image, 
            np.array([[10, 10], [90, 90]])
        )
        
        # 결과 검증
        self.assertEqual(result.shape, self.src_image.shape)
        self.assertEqual(result.dtype, self.src_image.dtype)
    
    def test_color_transfer_edge_cases(self):
        """엣지 케이스 테스트"""
        # 빈 이미지
        empty_src = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        empty_dst = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        
        with self.assertRaises(ValueError):
            self.processor.color_transfer(empty_src, empty_dst, np.array([[0, 0]]))
        
        # 크기가 다른 이미지
        small_src = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        large_dst = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = self.processor.color_transfer(
            small_src, large_dst, np.array([[10, 10], [90, 90]])
        )
        
        # 결과 검증 (크기가 조정되어야 함)
        self.assertEqual(result.shape, small_src.shape)
    
    def test_color_transfer_quality(self):
        """색상 전달 품질 테스트"""
        # 특정 패턴의 이미지 생성
        src_pattern = np.zeros((100, 100, 3), dtype=np.uint8)
        src_pattern[:, :, 0] = 255  # 빨간색
        
        dst_pattern = np.zeros((100, 100, 3), dtype=np.uint8)
        dst_pattern[:, :, 2] = 255  # 파란색
        
        result = self.processor.color_transfer(
            src_pattern, dst_pattern, 
            np.array([[10, 10], [90, 90]])
        )
        
        # 결과 검증
        self.assertEqual(result.shape, src_pattern.shape)
        
        # 색상이 변경되었는지 확인
        # (정확한 색상 매칭은 히스토그램 매칭 알고리즘에 따라 달라질 수 있음)
        self.assertFalse(np.array_equal(result, src_pattern))


if __name__ == '__main__':
    unittest.main()

