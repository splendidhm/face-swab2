# -*- coding: utf-8 -*-
"""
CLI 오케스트레이터 - 전체 파이프라인 제어
"""
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 한글 출력을 위한 인코딩 설정
import locale
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 모듈 임포트
from .common.utils import setup_logging, load_config, ensure_dir, get_timestamp
from .preprocess.preprocessor import Preprocessor
from .deepfacelab.trainer import DeepFaceLabTrainer
from .deepfacelab.inference import DeepFaceLabInference
from .fomm.trainer import FOMMTrainer
from .fomm.inference import FOMMInference
from .postprocess.color_correction import PostProcessor
from .postprocess.enhancement import ImageEnhancementPipeline
from .audio.audio_processor import AudioProcessor


class FaceSwapPipeline:
    """얼굴 교체 파이프라인 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # 컴포넌트 초기화
        self.preprocessor = Preprocessor(self.config.get('preprocessing', {}))
        self.postprocessor = PostProcessor(self.config.get('postprocessing', {}))
        self.audio_processor = AudioProcessor(self.config.get('audio', {}))
        self.enhancement_pipeline = ImageEnhancementPipeline(
            self.config.get('enhancement', {})
        )
    
    def preprocess_data(self, source_video: str, target_video: str, 
                       output_dir: str) -> Dict[str, Any]:
        """
        데이터 전처리
        
        Args:
            source_video: 소스 비디오
            target_video: 타겟 비디오
            output_dir: 출력 디렉토리
            
        Returns:
            dataset_info: 데이터셋 정보
        """
        self.logger.info("데이터 전처리 시작")
        
        ensure_dir(output_dir)
        
        # 얼굴 데이터셋 생성
        dataset_info = self.preprocessor.create_face_dataset(
            source_video, target_video, output_dir
        )
        
        self.logger.info("데이터 전처리 완료")
        return dataset_info
    
    def train_deepfacelab(self, dataset_info: Dict[str, Any], 
                         checkpoint_dir: str) -> None:
        """
        DeepFaceLab 모델 학습
        
        Args:
            dataset_info: 데이터셋 정보
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.logger.info("DeepFaceLab 모델 학습 시작")
        
        # 설정 로드
        dfl_config = load_config('configs/deepfacelab.yaml')
        
        # 트레이너 초기화
        trainer = DeepFaceLabTrainer(dfl_config)
        
        # 데이터 로더 준비
        train_loader, val_loader = trainer.prepare_data()
        
        # 학습
        trainer.train(train_loader, val_loader, checkpoint_dir)
        
        self.logger.info("DeepFaceLab 모델 학습 완료")
    
    def train_fomm(self, video_path: str, checkpoint_dir: str) -> None:
        """
        FOMM 모델 학습
        
        Args:
            video_path: 학습 비디오 경로
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.logger.info("FOMM 모델 학습 시작")
        
        # 설정 로드
        fomm_config = load_config('configs/fomm.yaml')
        
        # 트레이너 초기화
        trainer = FOMMTrainer(fomm_config)
        
        # 학습
        trainer.train(video_path, checkpoint_dir)
        
        self.logger.info("FOMM 모델 학습 완료")
    
    def inference_deepfacelab(self, source_video: str, target_video: str, 
                            output_video: str, checkpoint_path: str) -> None:
        """
        DeepFaceLab 모델 추론
        
        Args:
            source_video: 소스 비디오
            target_video: 타겟 비디오
            output_video: 출력 비디오
            checkpoint_path: 체크포인트 파일 경로
        """
        self.logger.info("DeepFaceLab 모델 추론 시작")
        
        # 설정 로드
        dfl_config = load_config('configs/deepfacelab.yaml')
        
        # 추론 엔진 초기화
        inference_engine = DeepFaceLabInference(dfl_config, checkpoint_path)
        
        # 얼굴 교체
        inference_engine.swap_face_video(source_video, target_video, output_video)
        
        self.logger.info("DeepFaceLab 모델 추론 완료")
    
    def inference_fomm(self, source_image: str, driving_video: str, 
                      output_video: str, checkpoint_path: str) -> None:
        """
        FOMM 모델 추론
        
        Args:
            source_image: 소스 이미지
            driving_video: 드라이빙 비디오
            output_video: 출력 비디오
            checkpoint_path: 체크포인트 파일 경로
        """
        self.logger.info("FOMM 모델 추론 시작")
        
        # 설정 로드
        fomm_config = load_config('configs/fomm.yaml')
        
        # 추론 엔진 초기화
        inference_engine = FOMMInference(fomm_config, checkpoint_path)
        
        # 이미지 로드
        import cv2
        source_img = cv2.imread(source_image)
        
        # 애니메이션
        inference_engine.animate_video(source_img, driving_video, output_video)
        
        self.logger.info("FOMM 모델 추론 완료")
    
    def postprocess_video(self, input_video: str, output_video: str, 
                         target_video: str = None) -> None:
        """
        비디오 후처리
        
        Args:
            input_video: 입력 비디오
            output_video: 출력 비디오
            target_video: 타겟 비디오 (색상 보정용)
        """
        self.logger.info("비디오 후처리 시작")
        
        if target_video:
            # 색상 보정 포함 후처리
            self.postprocessor.process_video(input_video, target_video, output_video)
        else:
            # 기본 후처리
            self.enhancement_pipeline.enhance_video(input_video, output_video)
        
        self.logger.info("비디오 후처리 완료")
    
    def process_audio(self, video_path: str, output_path: str, 
                     enhancement: bool = True) -> None:
        """
        오디오 처리
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 파일 경로
            enhancement: 향상 여부
        """
        self.logger.info("오디오 처리 시작")
        
        self.audio_processor.process_video_audio(video_path, output_path, enhancement)
        
        self.logger.info("오디오 처리 완료")
    
    def run_full_pipeline(self, source_video: str, target_video: str, 
                         output_video: str, model_type: str = 'deepfacelab') -> None:
        """
        전체 파이프라인 실행
        
        Args:
            source_video: 소스 비디오
            target_video: 타겟 비디오
            output_video: 출력 비디오
            model_type: 모델 타입 ('deepfacelab' 또는 'fomm')
        """
        self.logger.info(f"전체 파이프라인 시작: {model_type}")
        
        # 1. 데이터 전처리
        dataset_info = self.preprocess_data(source_video, target_video, 'data/processed')
        
        # 2. 모델 학습 (필요한 경우)
        if model_type == 'deepfacelab':
            self.train_deepfacelab(dataset_info, 'checkpoints/deepfacelab')
            checkpoint_path = 'checkpoints/deepfacelab/best.pth'
        elif model_type == 'fomm':
            self.train_fomm(target_video, 'checkpoints/fomm')
            checkpoint_path = 'checkpoints/fomm/best.pth'
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        # 3. 모델 추론
        if model_type == 'deepfacelab':
            self.inference_deepfacelab(source_video, target_video, 
                                     'temp_output.mp4', checkpoint_path)
        elif model_type == 'fomm':
            # FOMM의 경우 소스 이미지 필요
            import cv2
            source_img = cv2.imread('data/processed/source/aligned_face_000000.jpg')
            cv2.imwrite('temp_source.jpg', source_img)
            self.inference_fomm('temp_source.jpg', target_video, 
                              'temp_output.mp4', checkpoint_path)
        
        # 4. 후처리
        self.postprocess_video('temp_output.mp4', output_video, target_video)
        
        # 5. 오디오 처리
        self.process_audio(output_video, output_video, enhancement=True)
        
        # 임시 파일 정리
        Path('temp_output.mp4').unlink(missing_ok=True)
        Path('temp_source.jpg').unlink(missing_ok=True)
        
        self.logger.info("전체 파이프라인 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Face Swap Pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'inference', 'postprocess', 'audio', 'full'],
                       required=True, help='실행 모드')
    parser.add_argument('--source', type=str, help='소스 비디오/이미지 경로')
    parser.add_argument('--target', type=str, help='타겟 비디오 경로')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--checkpoint', type=str, help='체크포인트 파일 경로')
    parser.add_argument('--model', type=str, choices=['deepfacelab', 'fomm'],
                       default='deepfacelab', help='모델 타입')
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = FaceSwapPipeline(args.config)
    
    try:
        if args.mode == 'preprocess':
            if not args.source or not args.target or not args.output:
                print("preprocess 모드에서는 --source, --target, --output이 필요합니다")
                sys.exit(1)
            
            dataset_info = pipeline.preprocess_data(args.source, args.target, args.output)
            print(f"전처리 완료: {dataset_info}")
        
        elif args.mode == 'train':
            if not args.source or not args.output:
                print("train 모드에서는 --source, --output이 필요합니다")
                sys.exit(1)
            
            if args.model == 'deepfacelab':
                # DeepFaceLab 학습을 위한 데이터셋 정보 필요
                print("DeepFaceLab 학습을 위해서는 먼저 전처리를 수행해주세요")
                sys.exit(1)
            elif args.model == 'fomm':
                pipeline.train_fomm(args.source, args.output)
        
        elif args.mode == 'inference':
            if not args.source or not args.target or not args.output or not args.checkpoint:
                print("inference 모드에서는 --source, --target, --output, --checkpoint가 필요합니다")
                sys.exit(1)
            
            if args.model == 'deepfacelab':
                pipeline.inference_deepfacelab(args.source, args.target, args.output, args.checkpoint)
            elif args.model == 'fomm':
                pipeline.inference_fomm(args.source, args.target, args.output, args.checkpoint)
        
        elif args.mode == 'postprocess':
            if not args.source or not args.output:
                print("postprocess 모드에서는 --source, --output이 필요합니다")
                sys.exit(1)
            
            pipeline.postprocess_video(args.source, args.output, args.target)
        
        elif args.mode == 'audio':
            if not args.source or not args.output:
                print("audio 모드에서는 --source, --output이 필요합니다")
                sys.exit(1)
            
            pipeline.process_audio(args.source, args.output)
        
        elif args.mode == 'full':
            if not args.source or not args.target or not args.output:
                print("full 모드에서는 --source, --target, --output이 필요합니다")
                sys.exit(1)
            
            pipeline.run_full_pipeline(args.source, args.target, args.output, args.model)
        
        print("작업 완료!")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
