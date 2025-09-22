"""
비디오 입출력 유틸리티
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import imageio
from tqdm import tqdm


class VideoIO:
    """비디오 파일 읽기/쓰기 클래스"""
    
    def __init__(self):
        self.cap = None
        self.writer = None
    
    def read_video(self, video_path: str) -> Tuple[np.ndarray, dict]:
        """
        비디오 파일을 읽어서 프레임 배열과 메타데이터 반환
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            frames: 프레임 배열 (N, H, W, C)
            metadata: 비디오 메타데이터
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 메타데이터 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps if fps > 0 else 0
        }
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return np.array(frames), metadata
    
    def read_video_generator(self, video_path: str, sample_rate: int = 1) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        비디오를 프레임별로 생성기로 읽기
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 샘플링 비율 (1이면 모든 프레임)
            
        Yields:
            frame: 프레임 이미지
            frame_idx: 프레임 인덱스
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                yield frame, frame_idx
            
            frame_idx += 1
        
        cap.release()
    
    def write_video(self, frames: np.ndarray, output_path: str, fps: float = 30.0, 
                   codec: str = 'mp4v', quality: int = 90) -> None:
        """
        프레임 배열을 비디오 파일로 저장
        
        Args:
            frames: 프레임 배열 (N, H, W, C)
            output_path: 출력 파일 경로
            fps: 프레임 레이트
            codec: 코덱
            quality: 품질 (0-100)
        """
        if len(frames) == 0:
            raise ValueError("저장할 프레임이 없습니다")
        
        height, width = frames[0].shape[:2]
        
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # 비디오 라이터 생성
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"비디오 파일을 생성할 수 없습니다: {output_path}")
        
        # 프레임 저장
        for frame in tqdm(frames, desc="비디오 저장 중"):
            out.write(frame)
        
        out.release()
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      sample_rate: int = 1, format: str = 'jpg') -> List[str]:
        """
        비디오에서 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            sample_rate: 샘플링 비율
            format: 이미지 포맷
            
        Returns:
            extracted_files: 추출된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        for frame, frame_idx in self.read_video_generator(video_path, sample_rate):
            frame_filename = f"frame_{frame_idx:06d}.{format}"
            frame_path = output_path / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            extracted_files.append(str(frame_path))
        
        return extracted_files
    
    def get_video_info(self, video_path: str) -> dict:
        """
        비디오 파일 정보 반환
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            info: 비디오 정보 딕셔너리
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info

