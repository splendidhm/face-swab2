#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오디오 처리 스크립트
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
    """오디오 처리 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Processing')
    parser.add_argument('--input', type=str, required=True, help='입력 비디오 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 비디오 경로')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--enhancement', action='store_true', help='오디오 향상 적용')
    parser.add_argument('--lip-sync', type=str, help='립 싱크용 오디오 파일 경로')
    parser.add_argument('--tts', type=str, help='TTS용 텍스트')
    parser.add_argument('--tts-output', type=str, help='TTS 출력 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("오디오 처리 시작")
    
    try:
        # 파이프라인 초기화
        pipeline = FaceSwapPipeline(args.config)
        
        # 오디오 처리 실행
        if args.lip_sync:
            # 립 싱크 처리
            pipeline.audio_processor.create_lip_sync_video(
                args.input, args.lip_sync, args.output
            )
            logger.info("립 싱크 처리 완료")
            print(f"립 싱크 처리 완료: {args.output}")
        
        elif args.tts and args.tts_output:
            # TTS 처리
            tts_audio = pipeline.audio_processor.generate_tts_audio(
                args.tts, args.tts_output
            )
            logger.info(f"TTS 처리 완료: {tts_audio}")
            print(f"TTS 처리 완료: {tts_audio}")
        
        else:
            # 기본 오디오 처리
            pipeline.process_audio(args.input, args.output, args.enhancement)
            logger.info("오디오 처리 완료")
            print(f"오디오 처리 완료: {args.output}")
        
    except Exception as e:
        logger.error(f"오디오 처리 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
