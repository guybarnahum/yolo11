import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import numpy as np
import cv2
from typing import Optional, Tuple, List

class ConvLayer(nn.Module):
    """Basic convolutional layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False,
                              groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        
        self.conv1 = ConvLayer(in_channels, mid_channels, 1)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = LightConv3x3(mid_channels, mid_channels)
        self.conv2c = LightConv3x3(mid_channels, mid_channels)
        self.conv2d = LightConv3x3(mid_channels, mid_channels)
        
        # Channel attention
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels//16, 1),
            nn.BatchNorm2d(mid_channels//16),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//16, mid_channels, 1),
            nn.Sigmoid()
        )
        
        self.conv3 = ConvLayer(mid_channels, out_channels, 1)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)
        
        # Multi-scale feature fusion
        x2 = x2a + x2b + x2c + x2d
        x2 = self.gate(x2) * x2
        x3 = self.conv3(x2)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = x3 + identity
        return out

class OSNet(nn.Module):
    """OSNet with ImageNet architecture."""
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Conv1
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Conv2
        self.conv2 = nn.Sequential(
            OSBlock(64, 256),
            OSBlock(256, 256),
        )
        
        # Conv3
        self.conv3 = nn.Sequential(
            OSBlock(256, 384),
            OSBlock(384, 384),
        )
        
        # Conv4
        self.conv4 = nn.Sequential(
            OSBlock(384, 512),
            OSBlock(512, 512),
        )
        
        self.conv5 = ConvLayer(512, 512, 1)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.training:
            return x
        x = self.fc(x)
        return x

class OSNetEmbedder:
    """OSNet embedder compatible with DeepSort"""
    
    def __init__(self, model_path: str, use_cuda: bool = True, input_size: Tuple[int, int] = (256, 128)):
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        try:
            self.model = self._load_model(model_path)
            self.model = self.model.to(self.device).eval()
        except Exception as e:
            logging.error(f"Error loading OSNet model: {e}")
            raise

    def _load_model(self, model_path: str) -> nn.Module:
        try:
            # Create model for ImageNet weights
            model = OSNet(num_classes=1000)
            
            # Load pretrained weights
            state_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove module. prefix if present
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=True)
            logging.info("OSNet ImageNet model loaded successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error loading OSNet model: {e}")
            raise

    @torch.no_grad()
    def __call__(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract feature embeddings from image crops"""
        if not crops:
            return np.array([])
            
        # Preprocess images
        processed = []
        for crop in crops:
            if crop is None:
                continue
            crop = cv2.resize(crop, self.input_size)
            crop = crop.astype(np.float32) / 255.0
            crop = crop.transpose(2, 0, 1)
            processed.append(crop)
            
        if not processed:
            return np.array([])
            
        # Convert to tensor
        crops_tensor = torch.FloatTensor(processed).to(self.device)
        
        # Extract features
        features = self.model(crops_tensor)
        features = features.cpu().numpy()
        
        # Normalize
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        return features