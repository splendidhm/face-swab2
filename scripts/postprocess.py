#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
후처리 스크립트
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
    """후처리 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Swap Postprocessing')
    parser.add_argument('--input', type=str, required=True, help='입력 비디오 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 비디오 경로')
    parser.add_argument('--target', type=str, help='타겟 비디오 경로 (색상 보정용)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--enhancement', action='store_true', help='이미지 향상 적용')
    parser.add_argument('--color-correction', action='store_true', help='색상 보정 적용')
    parser.add_argument('--temporal-smoothing', action='store_true', help='시간적 스무딩 적용')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("후처리 시작")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 후처리 실행
        if args.target:
            # 색상 보정 포함 후처리
            pipeline.postprocess_video(args.input, args.output, args.target)
            logger.info("색상 보정 포함 후처리 완료")
        else:
            # 기본 후처리
            pipeline.postprocess_video(args.input, args.output)
            logger.info("기본 후처리 완료")
        
        print(f"후처리 완료: {args.output}")
        
    except Exception as e:
        logger.error(f"후처리 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
