#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리 스크립트
"""
import sys
import os
from pathlib import Path

# 한글 출력을 위한 인코딩 설정
import locale
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cli import FaceSwapPipeline
from src.common.utils import load_config, setup_logging


def main():
    """전처리 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Swap Preprocessing')
    parser.add_argument('--source', type=str, required=True, help='소스 비디오 경로')
    parser.add_argument('--target', type=str, required=True, help='타겟 비디오 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 디렉토리')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--sample-rate', type=int, default=1, help='샘플링 비율')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("전처리 시작")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 전처리 실행
        dataset_info = pipeline.preprocess_data(args.source, args.target, args.output)
        
        logger.info(f"전처리 완료: {dataset_info}")
        print(f"전처리 완료!")
        print(f"소스 얼굴: {dataset_info['source_faces']}개")
        print(f"타겟 얼굴: {dataset_info['target_faces']}개")
        print(f"출력 디렉토리: {args.output}")
        
    except Exception as e:
        logger.error(f"전처리 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
