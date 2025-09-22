"""
FOMM 모델 학습 클래스
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm
import json
from datetime import datetime

from .model import create_fomm_model, initialize_weights
from ..common.utils import setup_logging, ensure_dir, get_timestamp


class FOMMDataset(Dataset):
    """FOMM 데이터셋 클래스"""
    
    def __init__(self, video_path: str, transform=None, sequence_length: int = 5):
        """
        Args:
            video_path: 비디오 파일 경로
            transform: 이미지 변환 함수
            sequence_length: 시퀀스 길이
        """
        self.video_path = video_path
        self.transform = transform
        self.sequence_length = sequence_length
        
        # 비디오에서 프레임 추출
        from ..common.video_io import VideoIO
        video_io = VideoIO()
        
        self.frames = []
        for frame, _ in video_io.read_video_generator(video_path, sample_rate=1):
            self.frames.append(frame)
        
        self.logger = setup_logging()
        self.logger.info(f"로드된 프레임 수: {len(self.frames)}")
    
    def __len__(self):
        return max(0, len(self.frames) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        # 시퀀스 생성
        sequence = []
        for i in range(self.sequence_length):
            frame_idx = idx + i
            if frame_idx < len(self.frames):
                frame = self.frames[frame_idx]
                
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 크기 조정
                frame = cv2.resize(frame, (256, 256))
                
                # 정규화 (-1 to 1)
                frame = (frame.astype(np.float32) / 127.5) - 1.0
                
                # HWC to CHW
                frame = np.transpose(frame, (2, 0, 1))
                
                if self.transform:
                    frame = self.transform(frame)
                
                sequence.append(frame)
        
        # 첫 번째 프레임을 소스로, 나머지를 드라이빙으로 사용
        source = sequence[0]
        driving = sequence[1] if len(sequence) > 1 else sequence[0]
        
        return source, driving


class FOMMTrainer:
    """FOMM 모델 학습 클래스"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 학습 설정
        """
        self.config = config
        self.logger = setup_logging()
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = create_fomm_model(
            num_kp=config.get('num_kp', 10),
            num_channels=3
        ).to(self.device)
        
        # 옵티마이저 초기화
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
        
        # 판별자 옵티마이저
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()
        
        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 가중치 초기화
        initialize_weights(self.model)
    
    def prepare_data(self, video_path: str) -> DataLoader:
        """
        데이터 로더 준비
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            data_loader: 데이터 로더
        """
        # 데이터셋 생성
        dataset = FOMMDataset(
            video_path=video_path,
            sequence_length=self.config.get('sequence_length', 5)
        )
        
        # 데이터 로더 생성
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"데이터셋 크기: {len(dataset)}")
        
        return data_loader
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        한 에포크 학습
        
        Args:
            data_loader: 데이터 로더
            
        Returns:
            metrics: 학습 메트릭
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_gan_loss = 0.0
        total_kp_loss = 0.0
        
        for batch_idx, (source_imgs, driving_imgs) in enumerate(tqdm(data_loader, desc="학습 중")):
            source_imgs = source_imgs.to(self.device)
            driving_imgs = driving_imgs.to(self.device)
            
            # 생성자 학습
            self.optimizer.zero_grad()
            
            # 순전파
            generated_imgs, kp_source, kp_driving, jacobian = self.model(source_imgs, driving_imgs)
            
            # 재구성 손실
            recon_loss = self.l1_loss(generated_imgs, driving_imgs)
            
            # 키포인트 일관성 손실
            kp_loss = self.mse_loss(kp_source, kp_driving)
            
            # GAN 손실
            fake_pred = self.model.discriminator(generated_imgs)
            real_labels = torch.ones_like(fake_pred)
            gan_loss = self.gan_loss(fake_pred, real_labels)
            
            # 총 손실
            total_loss_batch = (
                recon_loss + 
                self.config.get('kp_weight', 0.1) * kp_loss +
                self.config.get('gan_weight', 0.1) * gan_loss
            )
            
            # 역전파
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 판별자 학습
            self.d_optimizer.zero_grad()
            
            # 실제 이미지
            real_pred = self.model.discriminator(driving_imgs)
            real_loss = self.gan_loss(real_pred, torch.ones_like(real_pred))
            
            # 가짜 이미지
            fake_pred = self.model.discriminator(generated_imgs.detach())
            fake_loss = self.gan_loss(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            self.d_optimizer.step()
            
            # 메트릭 업데이트
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_gan_loss += gan_loss.item()
            total_kp_loss += kp_loss.item()
        
        # 평균 계산
        metrics = {
            'total_loss': total_loss / len(data_loader),
            'recon_loss': total_recon_loss / len(data_loader),
            'gan_loss': total_gan_loss / len(data_loader),
            'kp_loss': total_kp_loss / len(data_loader)
        }
        
        return metrics
    
    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        검증 수행
        
        Args:
            data_loader: 데이터 로더
            
        Returns:
            metrics: 검증 메트릭
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        
        with torch.no_grad():
            for source_imgs, driving_imgs in data_loader:
                source_imgs = source_imgs.to(self.device)
                driving_imgs = driving_imgs.to(self.device)
                
                # 순전파
                generated_imgs, _, _, _ = self.model(source_imgs, driving_imgs)
                
                # 손실 계산
                recon_loss = self.l1_loss(generated_imgs, driving_imgs)
                total_loss += recon_loss.item()
                total_recon_loss += recon_loss.item()
        
        metrics = {
            'val_loss': total_loss / len(data_loader),
            'val_recon_loss': total_recon_loss / len(data_loader)
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       checkpoint_dir: str) -> None:
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에포크
            metrics: 메트릭
            checkpoint_dir: 체크포인트 디렉토리
        """
        ensure_dir(checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # 최신 체크포인트 저장
        checkpoint_path = Path(checkpoint_dir) / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if metrics.get('val_loss', float('inf')) < self.best_loss:
            self.best_loss = metrics['val_loss']
            best_path = Path(checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"최고 성능 모델 저장: {best_path}")
        
        # 주기적 체크포인트 저장
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_path = Path(checkpoint_dir) / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        
        self.logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
    
    def train(self, video_path: str, checkpoint_dir: str) -> None:
        """
        전체 학습 과정
        
        Args:
            video_path: 학습 비디오 경로
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.logger.info("FOMM 학습 시작")
        
        # 데이터 로더 준비
        data_loader = self.prepare_data(video_path)
        
        # 검증 데이터는 학습 데이터의 일부 사용
        val_size = len(data_loader.dataset) // 10
        train_size = len(data_loader.dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            data_loader.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 학습
            train_metrics = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = self.validate(val_loader)
            
            # 메트릭 로깅
            self.logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # 체크포인트 저장
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics, checkpoint_dir)
            
            # 학습률 스케줄링
            if epoch % 50 == 0 and epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.9
        
        self.logger.info("FOMM 학습 완료")

