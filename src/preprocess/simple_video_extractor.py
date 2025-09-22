# -*- coding: utf-8 -*-
"""
간단한 비디오 전처리 모듈 - 트래커 없이
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, List
import logging
from tqdm import tqdm
import yaml

from .face_detector_simple import SimpleFaceDetector


def iter_frames(video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """
    비디오 프레임 반복자
    
    Args:
        video_path: 비디오 파일 경로
        
    Yields:
        (frame_idx, frame_bgr): 프레임 인덱스와 BGR 프레임
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        yield frame_idx, frame
        frame_idx += 1
    
    cap.release()


class SimpleVideoPreprocessor:
    """간단한 비디오 전처리 클래스"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.face_detector = SimpleFaceDetector(
            min_detection_confidence=config.get('min_detection_confidence', 0.5)
        )
    
    def extract(self, video_path: str, sample_rate: int, out_dir: str) -> Dict:
        """
        비디오에서 얼굴 추출
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 샘플링 비율
            out_dir: 출력 디렉토리
            
        Returns:
            metadata: 추출된 메타데이터
        """
        self.logger.info(f"비디오 전처리 시작: {video_path}")
        
        # 출력 디렉토리 생성
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 메타데이터 초기화
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'sample_rate': sample_rate,
            'frames': {},
            'tracks': {}
        }
        
        # 얼굴 크롭 저장 디렉토리
        face_tracks_dir = out_path / 'face_tracks'
        face_tracks_dir.mkdir(exist_ok=True)
        
        # 프레임별 처리
        frame_count = 0
        detection_count = 0
        face_id_counter = 0
        
        for frame_idx, frame in tqdm(iter_frames(video_path), 
                                   desc="프레임 처리 중", 
                                   total=total_frames):
            
            frame_metadata = []
            
            # 샘플링에 따른 처리
            if frame_idx % sample_rate == 0:
                # 얼굴 검출
                detections = self.face_detector.detect(frame)
                detection_count += 1
                
                # 각 검출된 얼굴 처리
                for i, detection in enumerate(detections):
                    face_id = face_id_counter
                    face_id_counter += 1
                    
                    bbox = detection['bbox']
                    landmarks = np.array(detection['landmarks'])
                    quality_score = detection['quality_score']
                    
                    # 프레임 메타데이터에 추가
                    frame_metadata.append({
                        'face_id': face_id,
                        'bbox': bbox,
                        'landmarks': landmarks.tolist(),
                        'quality_score': quality_score
                    })
                    
                    # 얼굴 크롭 저장
                    if quality_score > 0.5:  # 품질이 좋은 경우만 저장
                        self._save_face_crop(frame, bbox, landmarks, face_id, 
                                           frame_idx, face_tracks_dir)
            
            # 프레임 메타데이터 저장
            if frame_metadata:
                metadata['frames'][str(frame_idx)] = frame_metadata
            
            frame_count += 1
        
        # 메타데이터 저장
        meta_path = out_path / 'source_video_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"전처리 완료: {frame_count} 프레임, {detection_count} 검출")
        self.logger.info(f"메타데이터 저장: {meta_path}")
        
        return metadata
    
    def _save_face_crop(self, frame: np.ndarray, bbox: List[int], 
                       landmarks: np.ndarray, face_id: int, 
                       frame_idx: int, face_tracks_dir: Path):
        """
        얼굴 크롭 저장
        
        Args:
            frame: 원본 프레임
            bbox: 바운딩 박스
            landmarks: 랜드마크
            face_id: 얼굴 ID
            frame_idx: 프레임 인덱스
            face_tracks_dir: 저장 디렉토리
        """
        x, y, w, h = bbox
        
        # 얼굴 크롭
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return
        
        # 얼굴 정렬
        aligned_face = self._align_face(face_crop, landmarks, bbox)
        
        # 저장 디렉토리 생성
        face_dir = face_tracks_dir / str(face_id)
        face_dir.mkdir(exist_ok=True)
        
        # 파일명 생성
        filename = f"frame_{frame_idx:06d}.png"
        filepath = face_dir / filename
        
        # 크롭된 얼굴 저장
        cv2.imwrite(str(filepath), aligned_face)
    
    def _align_face(self, face_crop: np.ndarray, landmarks: np.ndarray, 
                   bbox: List[int], output_size: int = 224) -> np.ndarray:
        """
        얼굴 정렬
        
        Args:
            face_crop: 얼굴 크롭
            landmarks: 랜드마크
            bbox: 바운딩 박스
            output_size: 출력 크기
            
        Returns:
            aligned_face: 정렬된 얼굴
        """
        # 간단한 리사이즈
        aligned_face = cv2.resize(face_crop, (output_size, output_size))
        return aligned_face


def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='간단한 비디오 얼굴 추출')
    parser.add_argument('--video', type=str, required=True, help='비디오 파일 경로')
    parser.add_argument('--sample-rate', type=int, default=3, help='샘플링 비율')
    parser.add_argument('--out', type=str, required=True, help='출력 디렉토리')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    
    args = parser.parse_args()
    
    # 설정 로드
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 전처리 실행
    preprocessor = SimpleVideoPreprocessor(config)
    
    # 얼굴 추출
    metadata = preprocessor.extract(args.video, args.sample_rate, args.out)
    
    print(f"전처리 완료: {args.out}")
    print(f"총 프레임: {metadata['total_frames']}")
    print(f"검출된 얼굴: {len(metadata['frames'])}개 프레임")


if __name__ == '__main__':
    main()
