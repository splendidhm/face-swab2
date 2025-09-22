# -*- coding: utf-8 -*-
"""
비디오 합성 모듈
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import subprocess
import tempfile

from .lightwarp import LightWarpProcessor
from .fomm_inference import FOMMProcessor


class VideoSynthesizer:
    """비디오 합성 클래스"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 처리기 초기화
        self.lightwarp_processor = None
        self.fomm_processor = None
        
        # 시간적 스무딩을 위한 버퍼
        self.mask_buffer = []
        self.smoothing_window = self.config.get('temporal_smoothing_window', 5)
    
    def synthesize_video(self, target_image_path: str, source_video_path: str, 
                        meta_json_path: str, mode: str, checkpoint: Optional[str] = None, 
                        out_path: str = None) -> str:
        """
        비디오 합성
        
        Args:
            target_image_path: 타겟 이미지 경로
            source_video_path: 소스 비디오 경로
            meta_json_path: 메타데이터 JSON 경로
            mode: 합성 모드 ('lightwarp' 또는 'fomm')
            checkpoint: 체크포인트 파일 경로 (FOMM용)
            out_path: 출력 경로
            
        Returns:
            output_path: 출력 파일 경로
        """
        self.logger.info(f"비디오 합성 시작: {mode} 모드")
        
        # 출력 경로 설정
        if out_path is None:
            out_path = f"./outputs/synthesized_{mode}.mp4"
        
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 디버그 디렉토리 생성
        debug_dir = output_path.parent / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        # 메타데이터 로드
        with open(meta_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 타겟 이미지 로드
        target_img = cv2.imread(target_image_path)
        if target_img is None:
            raise ValueError(f"타겟 이미지를 로드할 수 없습니다: {target_image_path}")
        
        # 처리기 초기화
        if mode == 'lightwarp':
            self.lightwarp_processor = LightWarpProcessor(self.config)
        elif mode == 'fomm':
            if checkpoint is None:
                raise ValueError("FOMM 모드에서는 체크포인트 파일이 필요합니다")
            self.fomm_processor = FOMMProcessor(checkpoint)
        else:
            raise ValueError(f"지원하지 않는 모드: {mode}")
        
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(source_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 임시 비디오 파일
        temp_video_path = str(output_path.with_suffix('.temp.mp4'))
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"비디오 라이터를 초기화할 수 없습니다: {temp_video_path}")
        
        # 프레임별 처리
        frame_count = 0
        cap = cv2.VideoCapture(source_video_path)
        
        for frame_idx in tqdm(range(total_frames), desc="프레임 합성 중"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 프레임의 얼굴 정보 가져오기
            frame_faces = metadata['frames'].get(str(frame_idx), [])
            
            if frame_faces:
                # 각 얼굴에 대해 처리
                for face_data in frame_faces:
                    face_id = face_data['face_id']
                    bbox = face_data['bbox']
                    landmarks = np.array(face_data['landmarks'])
                    
                    # 품질 필터링
                    quality_score = face_data.get('quality_score', 0.5)
                    if quality_score < 0.3:
                        continue
                    
                    # 얼굴 교체
                    if mode == 'lightwarp':
                        frame = self.lightwarp_processor.apply_target_to_frame(
                            frame, target_img, landmarks, mode
                        )
                    elif mode == 'fomm':
                        frame = self.fomm_processor.apply_target_to_frame(
                            frame, target_img, landmarks, mode
                        )
                
                # 시간적 스무딩 적용
                frame = self._apply_temporal_smoothing(frame, frame_faces)
            
            # 프레임 저장
            out.write(frame)
            
            # 디버그 이미지 저장 (옵션)
            if self.config.get('save_debug_images', False):
                debug_path = debug_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(debug_path), frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        # 오디오와 합성
        final_output = self._combine_with_audio(
            temp_video_path, source_video_path, str(output_path)
        )
        
        # 임시 파일 삭제
        Path(temp_video_path).unlink(missing_ok=True)
        
        self.logger.info(f"비디오 합성 완료: {final_output}")
        return final_output
    
    def _apply_temporal_smoothing(self, frame: np.ndarray, face_data: List[Dict]) -> np.ndarray:
        """
        시간적 스무딩 적용
        
        Args:
            frame: 현재 프레임
            face_data: 얼굴 데이터
            
        Returns:
            smoothed_frame: 스무딩된 프레임
        """
        if not self.config.get('temporal_smoothing', True):
            return frame
        
        # 마스크 버퍼 업데이트
        current_masks = []
        for face in face_data:
            landmarks = np.array(face['landmarks'])
            mask = self._create_face_mask(landmarks, frame.shape[:2])
            current_masks.append(mask)
        
        self.mask_buffer.append(current_masks)
        
        # 버퍼 크기 제한
        if len(self.mask_buffer) > self.smoothing_window:
            self.mask_buffer.pop(0)
        
        # 스무딩 적용
        if len(self.mask_buffer) >= 3:
            smoothed_frame = self._smooth_masks(frame)
            return smoothed_frame
        
        return frame
    
    def _smooth_masks(self, frame: np.ndarray) -> np.ndarray:
        """마스크 스무딩"""
        # 간단한 EMA 스무딩 구현
        # 실제로는 더 복잡한 시간적 일관성 알고리즘 사용 가능
        return frame
    
    def _create_face_mask(self, landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """얼굴 마스크 생성"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        # 컨벡스 헐 계산
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _combine_with_audio(self, video_path: str, source_video_path: str, 
                           output_path: str) -> str:
        """
        오디오와 합성
        
        Args:
            video_path: 비디오 파일 경로
            source_video_path: 원본 비디오 파일 경로
            output_path: 출력 파일 경로
            
        Returns:
            output_path: 최종 출력 파일 경로
        """
        try:
            # ffmpeg를 사용하여 오디오 추출 및 합성
            temp_audio_path = str(Path(output_path).with_suffix('.temp.wav'))
            
            # 원본 비디오에서 오디오 추출
            subprocess.run([
                'ffmpeg', '-i', source_video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2', temp_audio_path, '-y'
            ], check=True, capture_output=True)
            
            # 비디오와 오디오 합성
            subprocess.run([
                'ffmpeg', '-i', video_path, '-i', temp_audio_path, 
                '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 
                output_path, '-y'
            ], check=True, capture_output=True)
            
            # 임시 오디오 파일 삭제
            Path(temp_audio_path).unlink(missing_ok=True)
            
            self.logger.info(f"오디오 합성 완료: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"오디오 합성 실패: {e}")
            # 오디오 없이 비디오만 복사
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
        except FileNotFoundError:
            self.logger.warning("ffmpeg를 찾을 수 없습니다. 오디오 없이 비디오만 저장합니다.")
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
    
    def reset(self):
        """상태 리셋"""
        self.mask_buffer.clear()

