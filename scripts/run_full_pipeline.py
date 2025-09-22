#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 파이프라인 실행 스크립트
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
    """전체 파이프라인 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Full Face Swap Pipeline')
    parser.add_argument('--source', type=str, required=True, help='소스 비디오 경로')
    parser.add_argument('--target', type=str, required=True, help='타겟 비디오 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 비디오 경로')
    parser.add_argument('--model', type=str, choices=['deepfacelab', 'fomm'], 
                       default='deepfacelab', help='모델 타입')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--skip-preprocess', action='store_true', help='전처리 건너뛰기')
    parser.add_argument('--skip-train', action='store_true', help='학습 건너뛰기')
    parser.add_argument('--checkpoint', type=str, help='사전 훈련된 체크포인트 경로')
    parser.add_argument('--enhancement', action='store_true', help='향상 적용')
    parser.add_argument('--audio', action='store_true', help='오디오 처리 적용')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"전체 파이프라인 시작: {args.model}")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 전체 파이프라인 실행
        pipeline.run_full_pipeline(
            args.source, args.target, args.output, args.model
        )
        
        logger.info("전체 파이프라인 완료")
        print(f"전체 파이프라인 완료: {args.output}")
        
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
