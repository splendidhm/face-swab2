"""
DeepFaceLab 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class Encoder(nn.Module):
    """인코더 네트워크"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 인코더 레이어들
        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1)  # 256x256 -> 128x128
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 128x128 -> 64x64
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 64x64 -> 32x32
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 32x32 -> 16x16
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)  # 16x16 -> 8x8
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # 최종 레이어
        self.fc = nn.Linear(1024 * 8 * 8, latent_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class Decoder(nn.Module):
    """디코더 네트워크"""
    
    def __init__(self, latent_dim: int = 512, output_channels: int = 3):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 초기 레이어
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)
        
        # 디코더 레이어들
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 32x32 -> 64x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 64x64 -> 128x128
        self.deconv5 = nn.ConvTranspose2d(64, output_channels, 4, 2, 1)  # 128x128 -> 256x256
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 8, 8)
        
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.deconv1(x)))
        x = F.relu(self.bn3(self.deconv2(x)))
        x = F.relu(self.bn4(self.deconv3(x)))
        x = F.relu(self.bn5(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        
        return x


class SAE(nn.Module):
    """SAE (Source-Aligned Encoder) 모델"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super(SAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


class LIAE(nn.Module):
    """LIAE (Lightweight Identity-Aware Encoder) 모델"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super(LIAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        
        # Identity-aware 레이어
        self.identity_layer = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x):
        latent = self.encoder(x)
        identity_latent = self.identity_layer(latent)
        output = self.decoder(identity_latent)
        return output, latent, identity_latent


class DF(nn.Module):
    """DF (DeepFace) 모델"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super(DF, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        
        # 추가적인 특징 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        enhanced_latent = self.feature_extractor(latent)
        output = self.decoder(enhanced_latent)
        return output, latent, enhanced_latent


class Discriminator(nn.Module):
    """판별자 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1)
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


def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    모델 생성 팩토리 함수
    
    Args:
        model_type: 모델 타입 ('SAE', 'LIAE', 'DF')
        **kwargs: 모델 생성 인자
        
    Returns:
        model: 생성된 모델
    """
    if model_type == 'SAE':
        return SAE(**kwargs)
    elif model_type == 'LIAE':
        return LIAE(**kwargs)
    elif model_type == 'DF':
        return DF(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")


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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

