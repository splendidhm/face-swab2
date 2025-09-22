#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 추론 스크립트
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
    """추론 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Swap Inference')
    parser.add_argument('--model', type=str, choices=['deepfacelab', 'fomm'], 
                       required=True, help='모델 타입')
    parser.add_argument('--source', type=str, required=True, help='소스 비디오/이미지 경로')
    parser.add_argument('--target', type=str, required=True, help='타겟 비디오 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 비디오 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 파일 경로')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--sample-rate', type=int, default=1, help='샘플링 비율')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"{args.model} 모델 추론 시작")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 모델별 추론
        if args.model == 'deepfacelab':
            pipeline.inference_deepfacelab(args.source, args.target, args.output, args.checkpoint)
            logger.info("DeepFaceLab 추론 완료")
            print(f"DeepFaceLab 추론 완료: {args.output}")
        
        elif args.model == 'fomm':
            pipeline.inference_fomm(args.source, args.target, args.output, args.checkpoint)
            logger.info("FOMM 추론 완료")
            print(f"FOMM 추론 완료: {args.output}")
        
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
