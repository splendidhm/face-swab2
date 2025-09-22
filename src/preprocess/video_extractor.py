# -*- coding: utf-8 -*-
"""
비디오 전처리 모듈 - 얼굴 검출 및 트래킹
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, List
import logging
from tqdm import tqdm
import yaml

from .face_detector_simple import SimpleFaceDetector as FaceDetector
from .face_tracker import FaceTrackerManager


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


class VideoPreprocessor:
    """비디오 전처리 클래스"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('min_detection_confidence', 0.5)
        )
        
        self.tracker_manager = FaceTrackerManager(
            tracker_type=config.get('tracker_type', 'CSRT'),
            lost_face_timeout=config.get('lost_face_timeout', 30),
            iou_threshold=config.get('iou_threshold', 0.3)
        )
    
    def extract(self, video_path: str, sample_rate: int, out_dir: str) -> Dict:
        """
        비디오에서 얼굴 추출 및 트래킹
        
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
        
        for frame_idx, frame in tqdm(iter_frames(video_path), 
                                   desc="프레임 처리 중", 
                                   total=total_frames):
            
            frame_metadata = []
            
            # 샘플링에 따른 처리
            if frame_idx % sample_rate == 0:
                # 얼굴 검출
                detections = self.face_detector.detect(frame)
                detection_count += 1
            else:
                # 트래킹만 수행
                detections = []
            
            # 트래킹 업데이트
            tracked_faces = self.tracker_manager.update(detections, frame)
            
            # 각 트래킹된 얼굴 처리
            for face_data in tracked_faces:
                face_id = face_data['face_id']
                bbox = face_data['bbox']
                landmarks = np.array(face_data['landmarks'])
                quality_score = face_data['quality_score']
                
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
        
        # 트랙 정보 추가
        tracks = self.tracker_manager.get_face_tracks()
        metadata['tracks'] = {str(k): v for k, v in tracks.items()}
        
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
        # 랜드마크를 크롭 좌표계로 변환
        x, y, w, h = bbox
        crop_landmarks = landmarks - [x, y]
        
        # 기준 랜드마크 (정면 얼굴)
        reference_landmarks = self._get_reference_landmarks(output_size)
        
        # 호모그래피 행렬 계산
        h_matrix, _ = cv2.findHomography(
            crop_landmarks.astype(np.float32),
            reference_landmarks.astype(np.float32)
        )
        
        # 얼굴 정렬
        aligned_face = cv2.warpPerspective(
            face_crop, h_matrix, (output_size, output_size)
        )
        
        return aligned_face
    
    def _get_reference_landmarks(self, size: int) -> np.ndarray:
        """기준 랜드마크 생성"""
        # MediaPipe 468 랜드마크의 주요 포인트들
        # 눈, 코, 입의 좌표를 정면 얼굴 기준으로 설정
        center = size // 2
        
        reference_points = np.array([
            [center - 50, center - 30],  # 왼쪽 눈
            [center + 50, center - 30],  # 오른쪽 눈
            [center, center + 20],       # 코
            [center - 40, center + 60],  # 왼쪽 입
            [center + 40, center + 60], # 오른쪽 입
        ])
        
        return reference_points
    
    def visualize_tracking(self, video_path: str, output_path: str, 
                          sample_rate: int = 1) -> None:
        """
        트래킹 시각화 비디오 생성
        
        Args:
            video_path: 입력 비디오
            output_path: 출력 비디오
            sample_rate: 샘플링 비율
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 비디오 라이터
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        for frame_idx, frame in iter_frames(video_path):
            # 샘플링에 따른 처리
            if frame_idx % sample_rate == 0:
                detections = self.face_detector.detect(frame)
            else:
                detections = []
            
            # 트래킹 업데이트
            tracked_faces = self.tracker_manager.update(detections, frame)
            
            # 시각화
            vis_frame = self.face_detector.visualize_detections(frame, tracked_faces)
            
            # 얼굴 ID 표시
            for face_data in tracked_faces:
                bbox = face_data['bbox']
                face_id = face_data['face_id']
                x, y, w, h = bbox
                
                cv2.putText(vis_frame, f"ID: {face_id}", 
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(vis_frame)
        
        cap.release()
        out.release()
        
        self.logger.info(f"트래킹 시각화 완료: {output_path}")


def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='비디오 얼굴 추출')
    parser.add_argument('--video', type=str, required=True, help='비디오 파일 경로')
    parser.add_argument('--sample-rate', type=int, default=3, help='샘플링 비율')
    parser.add_argument('--out', type=str, required=True, help='출력 디렉토리')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='설정 파일')
    parser.add_argument('--tracker', type=str, choices=['CSRT', 'KCF', 'MOSSE'], 
                       default='CSRT', help='트래커 타입')
    parser.add_argument('--visualize', action='store_true', help='트래킹 시각화')
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 전처리 실행
    preprocessor = VideoPreprocessor(config)
    
    # 얼굴 추출
    metadata = preprocessor.extract(args.video, args.sample_rate, args.out)
    
    # 트래킹 시각화 (옵션)
    if args.visualize:
        vis_path = Path(args.out) / 'tracking_visualization.mp4'
        preprocessor.visualize_tracking(args.video, str(vis_path), args.sample_rate)
    
    print(f"전처리 완료: {args.out}")
    print(f"총 프레임: {metadata['total_frames']}")
    print(f"얼굴 트랙: {len(metadata['tracks'])}개")


if __name__ == '__main__':
    main()

