# -*- coding: utf-8 -*-
"""
얼굴 교체 추론 CLI
"""
import argparse
import sys
import os
from pathlib import Path
import logging
import yaml

# 한글 출력을 위한 인코딩 설정
import locale
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infer.video_synthesizer import VideoSynthesizer


def main():
    """메인 함수"""
    print("=" * 60)
    print("사용 전 대상자의 명시적 동의 필요 — 교육/연구 목적일 때만 사용하십시오.")
    print("=" * 60)
    print()
    
    parser = argparse.ArgumentParser(description='얼굴 교체 추론')
    parser.add_argument('--mode', type=str, choices=['lightwarp', 'fomm'], 
                       required=True, help='합성 모드')
    parser.add_argument('--target', type=str, required=True, 
                       help='타겟 이미지 경로')
    parser.add_argument('--video', type=str, required=True, 
                       help='소스 비디오 경로')
    parser.add_argument('--meta', type=str, required=True, 
                       help='메타데이터 JSON 경로')
    parser.add_argument('--out', type=str, required=True, 
                       help='출력 비디오 경로')
    parser.add_argument('--checkpoint', type=str, 
                       help='체크포인트 파일 경로 (FOMM용)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='설정 파일')
    parser.add_argument('--debug', action='store_true', 
                       help='디버그 이미지 저장')
    parser.add_argument('--smoothing', action='store_true', default=True, 
                       help='시간적 스무딩 적용')
    
    args = parser.parse_args()
    
    # 설정 로드
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    # 디버그 설정
    if args.debug:
        config['save_debug_images'] = True
    
    # 스무딩 설정
    config['temporal_smoothing'] = args.smoothing
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 입력 파일 검증
    if not Path(args.target).exists():
        print(f"타겟 이미지를 찾을 수 없습니다: {args.target}")
        sys.exit(1)
    
    if not Path(args.video).exists():
        print(f"소스 비디오를 찾을 수 없습니다: {args.video}")
        sys.exit(1)
    
    if not Path(args.meta).exists():
        print(f"메타데이터 파일을 찾을 수 없습니다: {args.meta}")
        sys.exit(1)
    
    # FOMM 모드에서 체크포인트 검증
    if args.mode == 'fomm':
        if not args.checkpoint:
            print("FOMM 모드에서는 --checkpoint 옵션이 필요합니다")
            sys.exit(1)
        
        if not Path(args.checkpoint).exists():
            print(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
            print("FOMM 체크포인트를 다운로드하고 지정된 경로에 배치하세요.")
            print("다운로드 링크: https://github.com/AliaksandrSiarohin/first-order-model")
            sys.exit(1)
    
    try:
        # 비디오 합성기 초기화
        synthesizer = VideoSynthesizer(config)
        
        # 비디오 합성 실행
        logger.info(f"얼굴 교체 시작: {args.mode} 모드")
        logger.info(f"타겟 이미지: {args.target}")
        logger.info(f"소스 비디오: {args.video}")
        logger.info(f"출력 경로: {args.out}")
        
        output_path = synthesizer.synthesize_video(
            target_image_path=args.target,
            source_video_path=args.video,
            meta_json_path=args.meta,
            mode=args.mode,
            checkpoint=args.checkpoint,
            out_path=args.out
        )
        
        print(f"\n얼굴 교체 완료!")
        print(f"출력 파일: {output_path}")
        
        if args.debug:
            debug_dir = Path(args.out).parent / 'debug'
            print(f"디버그 이미지: {debug_dir}")
        
    except Exception as e:
        logger.error(f"얼굴 교체 실패: {e}")
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

