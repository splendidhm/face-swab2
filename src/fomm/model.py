"""
First Order Motion Model (FOMM) 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class KPDetector(nn.Module):
    """키포인트 검출기"""
    
    def __init__(self, num_kp: int = 10, num_channels: int = 3):
        super(KPDetector, self).__init__()
        
        self.num_kp = num_kp
        self.num_channels = num_channels
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),  # 16 -> 8
            nn.ReLU(),
        )
        
        # 키포인트 예측 헤드
        self.kp_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_kp * 2, 1, 1, 0)  # x, y 좌표
        )
        
        # 자코비안 예측 헤드
        self.jacobian_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_kp * 4, 1, 1, 0)  # 2x2 변환 행렬
        )
        
    def forward(self, x):
        features = self.encoder(x)
        
        # 키포인트 예측
        kp = self.kp_head(features)
        kp = kp.view(kp.size(0), self.num_kp, 2)
        
        # 자코비안 예측
        jacobian = self.jacobian_head(features)
        jacobian = jacobian.view(jacobian.size(0), self.num_kp, 2, 2)
        
        return kp, jacobian


class Generator(nn.Module):
    """생성기 네트워크"""
    
    def __init__(self, num_kp: int = 10, num_channels: int = 3):
        super(Generator, self).__init__()
        
        self.num_kp = num_kp
        self.num_channels = num_channels
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),  # 16 -> 8
            nn.ReLU(),
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),  # 128 -> 256
            nn.ReLU(),
            nn.Conv2d(32, num_channels, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(self, x, kp_driving, kp_source, jacobian):
        # 인코딩
        features = self.encoder(x)
        
        # 키포인트 변환 적용
        transformed_features = self._apply_keypoint_transform(
            features, kp_driving, kp_source, jacobian
        )
        
        # 디코딩
        output = self.decoder(transformed_features)
        
        return output
    
    def _apply_keypoint_transform(self, features, kp_driving, kp_source, jacobian):
        """키포인트 변환 적용"""
        batch_size, channels, height, width = features.shape
        
        # 그리드 생성
        grid = self._create_grid(height, width, features.device)
        
        # 키포인트 변환 계산
        transformed_features = []
        
        for i in range(self.num_kp):
            # 키포인트 차이
            kp_diff = kp_driving[:, i] - kp_source[:, i]
            
            # 자코비안 적용
            jacobian_i = jacobian[:, i]
            
            # 변환된 그리드 계산
            transformed_grid = self._transform_grid(
                grid, kp_diff, jacobian_i, height, width
            )
            
            # 변환 적용
            transformed_feature = F.grid_sample(
                features, transformed_grid, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            
            transformed_features.append(transformed_feature)
        
        # 가중 평균
        weights = torch.softmax(torch.randn(batch_size, self.num_kp, 1, 1, 1).to(features.device), dim=1)
        output = sum(w * f for w, f in zip(weights.unbind(1), transformed_features))
        
        return output
    
    def _create_grid(self, height, width, device):
        """그리드 생성"""
        y = torch.linspace(-1, 1, height, device=device)
        x = torch.linspace(-1, 1, width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0)
    
    def _transform_grid(self, grid, kp_diff, jacobian, height, width):
        """그리드 변환"""
        batch_size = grid.size(0)
        
        # 키포인트 차이를 그리드 좌표로 변환
        kp_diff_norm = kp_diff * 2  # -1 to 1 범위로 정규화
        
        # 자코비안 적용
        jacobian_flat = jacobian.view(batch_size, 2, 2)
        
        # 변환된 그리드 계산
        grid_flat = grid.view(batch_size, -1, 2)
        transformed_grid = torch.bmm(
            grid_flat - kp_diff_norm.unsqueeze(1), 
            jacobian_flat.transpose(1, 2)
        ) + kp_diff_norm.unsqueeze(1)
        
        return transformed_grid.view(batch_size, height, width, 2)


class Discriminator(nn.Module):
    """판별자 네트워크"""
    
    def __init__(self, num_channels: int = 3):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        
        return x


class FOMM(nn.Module):
    """First Order Motion Model"""
    
    def __init__(self, num_kp: int = 10, num_channels: int = 3):
        super(FOMM, self).__init__()
        
        self.num_kp = num_kp
        self.num_channels = num_channels
        
        # 키포인트 검출기
        self.kp_detector = KPDetector(num_kp, num_channels)
        
        # 생성기
        self.generator = Generator(num_kp, num_channels)
        
        # 판별자
        self.discriminator = Discriminator(num_channels)
        
    def forward(self, source_image, driving_image):
        """
        Args:
            source_image: 소스 이미지
            driving_image: 드라이빙 이미지
            
        Returns:
            generated_image: 생성된 이미지
            kp_source: 소스 키포인트
            kp_driving: 드라이빙 키포인트
            jacobian: 자코비안
        """
        # 키포인트 검출
        kp_source, jacobian_source = self.kp_detector(source_image)
        kp_driving, jacobian_driving = self.kp_detector(driving_image)
        
        # 이미지 생성
        generated_image = self.generator(
            source_image, kp_driving, kp_source, jacobian_driving
        )
        
        return generated_image, kp_source, kp_driving, jacobian_driving
    
    def encode(self, image):
        """이미지 인코딩"""
        kp, jacobian = self.kp_detector(image)
        return kp, jacobian
    
    def decode(self, source_image, kp_driving, jacobian):
        """이미지 디코딩"""
        kp_source, _ = self.kp_detector(source_image)
        generated_image = self.generator(source_image, kp_driving, kp_source, jacobian)
        return generated_image


def create_fomm_model(num_kp: int = 10, num_channels: int = 3) -> FOMM:
    """
    FOMM 모델 생성
    
    Args:
        num_kp: 키포인트 개수
        num_channels: 채널 수
        
    Returns:
        model: FOMM 모델
    """
    return FOMM(num_kp, num_channels)


def initialize_weights(model: nn.Module):
    """모델 가중치 초기화"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

