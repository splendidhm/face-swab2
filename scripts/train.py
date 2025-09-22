#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 학습 스크립트
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
    """학습 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Swap Model Training')
    parser.add_argument('--model', type=str, choices=['deepfacelab', 'fomm'], 
                       required=True, help='모델 타입')
    parser.add_argument('--data', type=str, required=True, help='학습 데이터 경로')
    parser.add_argument('--output', type=str, required=True, help='체크포인트 출력 디렉토')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--epochs', type=int, default=1000, help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='학습률')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"{args.model} 모델 학습 시작")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 모델별 학습
        if args.model == 'deepfacelab':
            # DeepFaceLab 학습을 위한 데이터셋 정보 필요
            logger.info("DeepFaceLab 학습을 위해서는 먼저 전처리를 수행해주세요")
            print("DeepFaceLab 학습을 위해서는 먼저 전처리를 수행해주세요")
            print("사용법: python scripts/preprocess.py --source <source_video> --target <target_video> --output <output_dir>")
            sys.exit(1)
        
        elif args.model == 'fomm':
            # FOMM 학습
            pipeline.train_fomm(args.data, args.output)
            logger.info("FOMM 모델 학습 완료")
            print(f"FOMM 모델 학습 완료: {args.output}")
        
    except Exception as e:
        logger.error(f"학습 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
