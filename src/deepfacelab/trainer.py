"""
DeepFaceLab 모델 학습 클래스
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

from .model import create_model, initialize_weights, Discriminator
from ..common.utils import setup_logging, ensure_dir, get_timestamp


class FaceDataset(Dataset):
    """얼굴 데이터셋 클래스"""
    
    def __init__(self, source_dir: str, target_dir: str, transform=None):
        """
        Args:
            source_dir: 소스 얼굴 이미지 디렉토리
            target_dir: 타겟 얼굴 이미지 디렉토리
            transform: 이미지 변환 함수
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        
        # 이미지 파일 목록 생성
        self.source_files = list(self.source_dir.glob("*.jpg")) + list(self.source_dir.glob("*.png"))
        self.target_files = list(self.target_dir.glob("*.jpg")) + list(self.target_dir.glob("*.png"))
        
        # 파일명으로 매칭
        self.pairs = []
        for source_file in self.source_files:
            target_file = self.target_dir / source_file.name
            if target_file.exists():
                self.pairs.append((source_file, target_file))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        source_path, target_path = self.pairs[idx]
        
        # 이미지 로드
        source_img = cv2.imread(str(source_path))
        target_img = cv2.imread(str(target_path))
        
        if source_img is None or target_img is None:
            # 빈 이미지 반환
            empty_img = np.zeros((256, 256, 3), dtype=np.uint8)
            return empty_img, empty_img
        
        # BGR to RGB
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # 정규화 (-1 to 1)
        source_img = (source_img.astype(np.float32) / 127.5) - 1.0
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0
        
        # HWC to CHW
        source_img = np.transpose(source_img, (2, 0, 1))
        target_img = np.transpose(target_img, (2, 0, 1))
        
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        return source_img, target_img


class DeepFaceLabTrainer:
    """DeepFaceLab 모델 학습 클래스"""
    
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
        self.model = create_model(
            model_type=config['model']['name'],
            input_channels=3,
            latent_dim=config['model'].get('latent_dim', 512)
        ).to(self.device)
        
        # 판별자 초기화 (GAN 학습 시)
        if config.get('use_gan', False):
            self.discriminator = Discriminator().to(self.device)
            self.d_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=config['training']['learning_rate'],
                betas=(0.5, 0.999)
            )
        else:
            self.discriminator = None
        
        # 옵티마이저 초기화
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 가중치 초기화
        initialize_weights(self.model)
        if self.discriminator:
            initialize_weights(self.discriminator)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        데이터 로더 준비
        
        Returns:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
        """
        # 데이터셋 생성
        train_dataset = FaceDataset(
            source_dir=self.config['data']['source_dir'],
            target_dir=self.config['data']['target_dir']
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 검증 데이터는 학습 데이터의 일부 사용
        val_size = len(train_dataset) // 10
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        한 에포크 학습
        
        Args:
            train_loader: 학습 데이터 로더
            
        Returns:
            metrics: 학습 메트릭
        """
        self.model.train()
        if self.discriminator:
            self.discriminator.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_gan_loss = 0.0
        
        for batch_idx, (source_imgs, target_imgs) in enumerate(tqdm(train_loader, desc="학습 중")):
            source_imgs = source_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)
            
            # 생성자 학습
            self.optimizer.zero_grad()
            
            # 순전파
            if self.config['model']['name'] == 'SAE':
                output, latent = self.model(source_imgs)
            elif self.config['model']['name'] == 'LIAE':
                output, latent, identity_latent = self.model(source_imgs)
            elif self.config['model']['name'] == 'DF':
                output, latent, enhanced_latent = self.model(source_imgs)
            
            # 재구성 손실
            recon_loss = self.l1_loss(output, target_imgs)
            
            # GAN 손실 (사용 시)
            gan_loss = 0.0
            if self.discriminator:
                fake_pred = self.discriminator(output)
                gan_loss = -torch.mean(torch.log(fake_pred + 1e-8))
            
            # 총 손실
            total_loss_batch = recon_loss + self.config.get('gan_weight', 0.0) * gan_loss
            
            # 역전파
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 판별자 학습 (GAN 사용 시)
            if self.discriminator:
                self.d_optimizer.zero_grad()
                
                # 실제 이미지
                real_pred = self.discriminator(target_imgs)
                real_loss = -torch.mean(torch.log(real_pred + 1e-8))
                
                # 가짜 이미지
                fake_pred = self.discriminator(output.detach())
                fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()
            
            # 메트릭 업데이트
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            if self.discriminator:
                total_gan_loss += gan_loss.item()
        
        # 평균 계산
        metrics = {
            'total_loss': total_loss / len(train_loader),
            'recon_loss': total_recon_loss / len(train_loader),
            'gan_loss': total_gan_loss / len(train_loader) if self.discriminator else 0.0
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        검증 수행
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            metrics: 검증 메트릭
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        
        with torch.no_grad():
            for source_imgs, target_imgs in val_loader:
                source_imgs = source_imgs.to(self.device)
                target_imgs = target_imgs.to(self.device)
                
                # 순전파
                if self.config['model']['name'] == 'SAE':
                    output, latent = self.model(source_imgs)
                elif self.config['model']['name'] == 'LIAE':
                    output, latent, identity_latent = self.model(source_imgs)
                elif self.config['model']['name'] == 'DF':
                    output, latent, enhanced_latent = self.model(source_imgs)
                
                # 손실 계산
                recon_loss = self.l1_loss(output, target_imgs)
                total_loss += recon_loss.item()
                total_recon_loss += recon_loss.item()
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_recon_loss': total_recon_loss / len(val_loader)
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
            'metrics': metrics,
            'config': self.config
        }
        
        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['d_optimizer_state_dict'] = self.d_optimizer.state_dict()
        
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
        
        if self.discriminator and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        
        self.logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              checkpoint_dir: str) -> None:
        """
        전체 학습 과정
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.logger.info("학습 시작")
        
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
        
        self.logger.info("학습 완료")

