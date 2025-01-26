import logging

def detect_pose_from_bbox(bbox):
    
    # Bounding box dimensions
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
 
    # Rough yaw estimation based on aspect ratio
    aspect_ratio = width / height
    
    # Typical car aspect ratios by orientation
    if aspect_ratio < 1.2:  # Nearly square (front/back view)
        yaw = 0  # or 180 degrees
    elif aspect_ratio > 2:  # Wide (side view)
        yaw = 90 # or 270 degrees
    else :
        yaw = 45 
    return yaw


import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models
from torchvision.models import ResNet18_Weights
from torchvision.models import mobilenet_v3_large


from utils import cuda_device

# Load a pre-trained ResNet model
class YawEstimatorWithPretrainedModel(nn.Module):
    def __init__(self):
        super(YawEstimatorWithPretrainedModel, self).__init__()
        # self.feature_extractor = torchvision_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = mobilenet_v3_large(weights="IMAGENET1K_V1")
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
        # Replace the final layer with a regression head
        num_features = self.feature_extractor.classifier[0].in_features
        self.feature_extractor.classifier = nn.Identity()   # Remove the final classification layer
        self.regressor = nn.Linear(num_features, 1) # Add a regression layer for yaw estimation
        
        '''
            self.regressor = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Tanh()  # Limit output to [-1, 1]
            )
        '''

    def forward(self, x):
        features = self.feature_extractor(x)
        yaw = self.regressor(features) 
        return yaw

# Initialize the model
device     = cuda_device()
yaw_model  = YawEstimatorWithPretrainedModel()
yaw_model  = yaw_model.to(device)
yaw_model.eval() # Freeze the entire model (if no training is allowed)

# Transform for input frames already normalized PyTorch Tensor type
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Optional if resizing is needed
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_pose(detection, car_frame):
    global yaw_model

    # Convert car_frame to PyTorch Tensor
    model_input = car_frame.astype(np.float32) / 255.0  # Scale to [0, 1]
    model_input = torch.from_numpy(model_input).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # If needed, resize the image to the input size of the model (224x224)
    resize = transforms.Resize((224, 224))
    model_input = transform(model_input)  # Add batch dimension for resize

    # Normalize the tensor
    model_input = transform(model_input.squeeze(0))  # Apply normalization

    # Add batch dimension again for inference
    model_input = model_input.unsqueeze(0)

    with torch.no_grad():
        yaw = yaw_model(model_input)  # Pass through the model

    return yaw.item()


