"""
오디오 처리 모듈
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import cv2
from pydub import AudioSegment
import ffmpeg
from scipy import signal
from scipy.signal import butter, filtfilt


class AudioExtractor:
    """오디오 추출 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Args:
            sample_rate: 샘플 레이트
        """
        self.sample_rate = sample_rate
    
    def extract_from_video(self, video_path: str, output_path: str) -> str:
        """
        비디오에서 오디오 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 오디오 파일 경로
            
        Returns:
            output_path: 출력 파일 경로
        """
        try:
            # ffmpeg를 사용하여 오디오 추출
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ar=self.sample_rate)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except Exception as e:
            raise ValueError(f"오디오 추출 실패: {e}")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        오디오 파일 로드
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            audio_data: 오디오 데이터
            sample_rate: 샘플 레이트
        """
        audio_data, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        return audio_data, sample_rate
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, 
                   sample_rate: int = None) -> None:
        """
        오디오 파일 저장
        
        Args:
            audio_data: 오디오 데이터
            output_path: 출력 파일 경로
            sample_rate: 샘플 레이트
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        sf.write(output_path, audio_data, sample_rate)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        오디오 길이 가져오기
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            duration: 오디오 길이 (초)
        """
        audio_data, sample_rate = self.load_audio(audio_path)
        return len(audio_data) / sample_rate


class AudioSynchronizer:
    """오디오 동기화 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Args:
            sample_rate: 샘플 레이트
        """
        self.sample_rate = sample_rate
    
    def synchronize_audio_video(self, audio_path: str, video_path: str, 
                              output_path: str, offset: float = 0.0) -> None:
        """
        오디오와 비디오 동기화
        
        Args:
            audio_path: 오디오 파일 경로
            video_path: 비디오 파일 경로
            output_path: 출력 비디오 파일 경로
            offset: 오프셋 (초)
        """
        try:
            # ffmpeg를 사용하여 오디오와 비디오 합성
            if offset != 0:
                # 오프셋 적용
                (
                    ffmpeg
                    .input(video_path)
                    .input(audio_path)
                    .output(output_path, vcodec='copy', acodec='aac', 
                           **{'itsoffset': offset})
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                (
                    ffmpeg
                    .input(video_path)
                    .input(audio_path)
                    .output(output_path, vcodec='copy', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
        except Exception as e:
            raise ValueError(f"오디오-비디오 동기화 실패: {e}")
    
    def align_audio_with_frames(self, audio_data: np.ndarray, 
                               frame_count: int, fps: float) -> np.ndarray:
        """
        오디오를 프레임과 정렬
        
        Args:
            audio_data: 오디오 데이터
            frame_count: 프레임 수
            fps: 프레임 레이트
            
        Returns:
            aligned_audio: 정렬된 오디오
        """
        # 비디오 길이 계산
        video_duration = frame_count / fps
        
        # 오디오 길이 계산
        audio_duration = len(audio_data) / self.sample_rate
        
        if audio_duration > video_duration:
            # 오디오가 더 긴 경우 자르기
            target_length = int(video_duration * self.sample_rate)
            aligned_audio = audio_data[:target_length]
        elif audio_duration < video_duration:
            # 오디오가 더 짧은 경우 패딩
            target_length = int(video_duration * self.sample_rate)
            padding_length = target_length - len(audio_data)
            padding = np.zeros(padding_length)
            aligned_audio = np.concatenate([audio_data, padding])
        else:
            aligned_audio = audio_data
        
        return aligned_audio


class AudioEnhancer:
    """오디오 향상 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Args:
            sample_rate: 샘플 레이트
        """
        self.sample_rate = sample_rate
    
    def enhance_audio(self, audio_data: np.ndarray, 
                     enhancement_type: str = 'normalize') -> np.ndarray:
        """
        오디오 향상
        
        Args:
            audio_data: 오디오 데이터
            enhancement_type: 향상 타입
            
        Returns:
            enhanced_audio: 향상된 오디오
        """
        if enhancement_type == 'normalize':
            return self._normalize_audio(audio_data)
        elif enhancement_type == 'noise_reduction':
            return self._reduce_noise(audio_data)
        elif enhancement_type == 'equalize':
            return self._equalize_audio(audio_data)
        elif enhancement_type == 'compress':
            return self._compress_audio(audio_data)
        else:
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """오디오 정규화"""
        # 최대값으로 정규화
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized = audio_data / max_val
        else:
            normalized = audio_data
        
        return normalized
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """노이즈 감소"""
        # 스펙트럼 차감을 사용한 노이즈 감소
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 노이즈 추정 (처음 0.5초)
        noise_frames = int(0.5 * self.sample_rate / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # 스펙트럼 차감
        alpha = 2.0
        beta = 0.01
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # ISTFT로 복원
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def _equalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """오디오 이퀄라이징"""
        # 멜 스펙트로그램 계산
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=self.sample_rate, n_mels=128
        )
        
        # 로그 변환
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 이퀄라이징 (간단한 구현)
        equalized_spec = log_mel_spec * 1.2  # 고주파 부스트
        
        # 다시 오디오로 변환
        equalized_mel_spec = librosa.db_to_power(equalized_spec)
        equalized_audio = librosa.feature.inverse.mel_to_audio(
            equalized_mel_spec, sr=self.sample_rate
        )
        
        return equalized_audio
    
    def _compress_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """오디오 압축"""
        # 다이나믹 레인지 압축
        threshold = 0.5
        ratio = 4.0
        
        compressed = np.where(
            np.abs(audio_data) > threshold,
            np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) / ratio),
            audio_data
        )
        
        return compressed


class LipSyncProcessor:
    """립 싱크 처리 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Args:
            sample_rate: 샘플 레이트
        """
        self.sample_rate = sample_rate
    
    def extract_audio_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        오디오에서 특징 추출
        
        Args:
            audio_data: 오디오 데이터
            
        Returns:
            features: 추출된 특징
        """
        # MFCC 특징 추출
        mfcc = librosa.feature.mfcc(
            y=audio_data, sr=self.sample_rate, n_mfcc=13
        )
        
        # 델타 특징 계산
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 특징 결합
        features = np.vstack([mfcc, delta, delta2])
        
        return features.T  # (time, features)
    
    def align_audio_with_visual(self, audio_data: np.ndarray, 
                               visual_frames: int, fps: float) -> np.ndarray:
        """
        오디오와 비주얼 정렬
        
        Args:
            audio_data: 오디오 데이터
            visual_frames: 비주얼 프레임 수
            fps: 프레임 레이트
            
        Returns:
            aligned_audio: 정렬된 오디오
        """
        # 비주얼 길이 계산
        visual_duration = visual_frames / fps
        
        # 오디오 길이 계산
        audio_duration = len(audio_data) / self.sample_rate
        
        if audio_duration > visual_duration:
            # 오디오가 더 긴 경우 자르기
            target_length = int(visual_duration * self.sample_rate)
            aligned_audio = audio_data[:target_length]
        elif audio_duration < visual_duration:
            # 오디오가 더 짧은 경우 패딩
            target_length = int(visual_duration * self.sample_rate)
            padding_length = target_length - len(audio_data)
            padding = np.zeros(padding_length)
            aligned_audio = np.concatenate([audio_data, padding])
        else:
            aligned_audio = audio_data
        
        return aligned_audio
    
    def create_audio_visual_sync(self, audio_path: str, video_path: str, 
                              output_path: str) -> None:
        """
        오디오-비주얼 동기화 생성
        
        Args:
            audio_path: 오디오 파일 경로
            video_path: 비디오 파일 경로
            output_path: 출력 파일 경로
        """
        # 오디오 로드
        audio_data, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        
        # 비디오 정보 가져오기
        from ..common.video_io import VideoIO
        video_io = VideoIO()
        video_info = video_io.get_video_info(video_path)
        
        # 오디오 정렬
        aligned_audio = self.align_audio_with_visual(
            audio_data, video_info['frame_count'], video_info['fps']
        )
        
        # 임시 오디오 파일 저장
        temp_audio_path = str(Path(output_path).with_suffix('.wav'))
        sf.write(temp_audio_path, aligned_audio, sample_rate)
        
        # 오디오와 비디오 합성
        synchronizer = AudioSynchronizer(sample_rate)
        synchronizer.synchronize_audio_video(temp_audio_path, video_path, output_path)
        
        # 임시 파일 삭제
        Path(temp_audio_path).unlink(missing_ok=True)


class TTSProcessor:
    """TTS (Text-to-Speech) 처리 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Args:
            sample_rate: 샘플 레이트
        """
        self.sample_rate = sample_rate
    
    def text_to_speech(self, text: str, output_path: str, 
                      voice: str = 'default') -> str:
        """
        텍스트를 음성으로 변환
        
        Args:
            text: 입력 텍스트
            output_path: 출력 파일 경로
            voice: 음성 타입
            
        Returns:
            output_path: 출력 파일 경로
        """
        try:
            # 간단한 TTS 구현 (실제로는 gTTS, pyttsx3 등 사용)
            # 여기서는 더미 구현
            duration = len(text) * 0.1  # 대략적인 길이 계산
            audio_data = np.random.normal(0, 0.1, int(duration * self.sample_rate))
            
            # 오디오 저장
            sf.write(output_path, audio_data, self.sample_rate)
            
            return output_path
        except Exception as e:
            raise ValueError(f"TTS 변환 실패: {e}")
    
    def synthesize_speech(self, text: str, output_path: str, 
                         language: str = 'ko') -> str:
        """
        음성 합성
        
        Args:
            text: 입력 텍스트
            output_path: 출력 파일 경로
            language: 언어 코드
            
        Returns:
            output_path: 출력 파일 경로
        """
        # 실제 TTS 엔진 사용 (gTTS, pyttsx3, etc.)
        # 여기서는 더미 구현
        return self.text_to_speech(text, output_path)


class AudioProcessor:
    """오디오 처리 통합 클래스"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: 오디오 처리 설정
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        
        # 컴포넌트 초기화
        self.extractor = AudioExtractor(self.sample_rate)
        self.synchronizer = AudioSynchronizer(self.sample_rate)
        self.enhancer = AudioEnhancer(self.sample_rate)
        self.lip_sync = LipSyncProcessor(self.sample_rate)
        self.tts = TTSProcessor(self.sample_rate)
    
    def process_video_audio(self, video_path: str, output_path: str, 
                           enhancement: bool = True) -> str:
        """
        비디오 오디오 처리
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 파일 경로
            enhancement: 향상 여부
            
        Returns:
            output_path: 출력 파일 경로
        """
        # 오디오 추출
        temp_audio_path = str(Path(output_path).with_suffix('.wav'))
        self.extractor.extract_from_video(video_path, temp_audio_path)
        
        # 오디오 로드
        audio_data, sample_rate = self.extractor.load_audio(temp_audio_path)
        
        # 오디오 향상
        if enhancement:
            audio_data = self.enhancer.enhance_audio(audio_data, 'normalize')
            audio_data = self.enhancer.enhance_audio(audio_data, 'noise_reduction')
        
        # 향상된 오디오 저장
        self.extractor.save_audio(audio_data, temp_audio_path, sample_rate)
        
        # 비디오와 오디오 합성
        self.synchronizer.synchronize_audio_video(temp_audio_path, video_path, output_path)
        
        # 임시 파일 삭제
        Path(temp_audio_path).unlink(missing_ok=True)
        
        return output_path
    
    def create_lip_sync_video(self, source_video: str, target_audio: str, 
                            output_video: str) -> str:
        """
        립 싱크 비디오 생성
        
        Args:
            source_video: 소스 비디오
            target_audio: 타겟 오디오
            output_video: 출력 비디오
            
        Returns:
            output_video: 출력 비디오 경로
        """
        self.lip_sync.create_audio_visual_sync(target_audio, source_video, output_video)
        return output_video
    
    def generate_tts_audio(self, text: str, output_path: str) -> str:
        """
        TTS 오디오 생성
        
        Args:
            text: 입력 텍스트
            output_path: 출력 파일 경로
            
        Returns:
            output_path: 출력 파일 경로
        """
        return self.tts.text_to_speech(text, output_path)

