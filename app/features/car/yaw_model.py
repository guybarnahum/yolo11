import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models
from torchvision.models import ResNet18_Weights
from torchvision.models import mobilenet_v3_large

import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from utils import cuda_device

device = cuda_device()

class CarYawEstimator(nn.Module):
    def __init__(self, num_bins=8):
        super(CarYawEstimator, self).__init__()
        # Load pretrained MobileNetV2
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1")
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_bins)
        )

        # Define bin centers (in degrees)
        self.bin_centers = torch.tensor([
            0,    # front
            45,   # front-right
            90,   # right
            135,  # back-right
            180,  # back
            225,  # back-left
            270,  # left
            315   # front-left
        ])

        # Define transform as part of the model
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.backbone(x)

    def preprocess(self, image):
        """
        Apply the transform to a given image.
        :param image: PIL image or NumPy array
        :return: Preprocessed tensor ready for inference
        """
        return self.transform(image).unsqueeze(0)  # Add batch dimension


def angle_to_bin(angle, num_bins=8):
    """
    Convert angle in degrees to bin index.
    Handles angles as tensors or scalars.
    """
    # Convert tensor to scalar if necessary
    if isinstance(angle, torch.Tensor):
        angle = angle.item()  # Extract scalar value from tensor
    
    angle = angle % 360         # Normalize angle to [0, 360)
    bin_size = 360 / num_bins   # Each bin covers 360/num_bins degrees
    bin_idx = int(round(angle / bin_size) % num_bins)  # Find nearest bin

    return bin_idx

def angle_to_bin_vectorized(angles, num_bins=8):
    """
    Vectorized version of angle_to_bin for PyTorch tensors.
    Converts angles in degrees to bin indices.
    """
    angles = angles % 360                # Normalize to [0, 360)
    bin_size = 360 / num_bins            # Bin size
    bin_indices = torch.round(angles / bin_size) % num_bins  # Compute bin indices
    return bin_indices.to(torch.long)    # Return as integer tensor


def bin_to_angle(bin_idx, num_bins=8):
    '''
    Convert bin index to angle in degrees
    '''
    return (bin_idx * 360 / num_bins) % 360


def setup_yaw_model(model_wts_path=None):
    # Initialize the model
    device     = cuda_device()
    yaw_model  = CarYawEstimator()
    
    if model_wts_path : 
        yaw_model.load_state_dict(torch.load(model_wts_path, map_location=device))

    yaw_model  = yaw_model.to(device)
    yaw_model.eval() # Freeze the entire model (if no training is allowed)

    return yaw_model

def predict_yaw( yaw_model, car_frame):
    global device
    
    car_frame_tensor = yaw_model.preprocess(car_frame).to(device)

    with torch.no_grad():
        predictions = yaw_model(car_frame_tensor)
        probs = F.softmax(predictions, dim=1)
        max_bin = torch.argmax(probs, dim=1)
        predicted_angle = yaw_model.bin_centers[max_bin].item()
        confidence = torch.max(probs, dim=1)[0].item()

    # print(f"Predicted Yaw Angle: {predicted_angle} degrees, Confidence: {confidence * 100:.2f}%")
    return predicted_angle, confidence


'''
Training CarYawEstimator

    dataset/
    0/  # Images of cars facing 0 degrees
    45/ # Images of cars facing 45 degrees
    ...
    315/ # Images of cars facing 315 degrees

'''
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CarYawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Iterate through angle directories
        for angle_dir in os.listdir(root_dir):
            angle_path = os.path.join(root_dir, angle_dir)
            if not os.path.isdir(angle_path):
                continue
            
            # Parse the angle as integer
            angle = int(angle_dir)
            for img_name in os.listdir(angle_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_name.startswith('.'):
                    img_path = os.path.join(angle_path, img_name)
                    self.samples.append((img_path, angle))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, angle = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, angle


def compute_loss(predictions, targets):
    """
    Custom loss function that handles circular nature of angles
    predictions: tensor of shape (batch_size, num_bins)
    targets: tensor of shape (batch_size,) containing bin indices
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(predictions, targets)
    
    # Add circular consistency loss
    softmax_preds = F.softmax(predictions, dim=1)
    batch_size = predictions.size(0)
    num_bins = predictions.size(1)
    
    # Calculate angular difference between adjacent bins
    circular_loss = 0
    for i in range(num_bins):
        next_bin = (i + 1) % num_bins
        diff = torch.abs(softmax_preds[:, i] - softmax_preds[:, next_bin])
        circular_loss += diff.mean()
    
    # Combine losses
    total_loss = ce_loss + 0.1 * circular_loss
    return total_loss

def train_step(model, optimizer, images, angles):
    optimizer.zero_grad()
    
    # Convert target angles to bins
    target_bins = angle_to_bin_vectorized(angles)
    
    # Forward pass
    predictions = model(images)
    
    # Compute loss
    loss = compute_loss(predictions, target_bins)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_epochs_loop(model, optimizer, dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for images, angles in dataloader:  # Iterate through DataLoader
            images = images.to(device)
            angles = angles.to(device)

            # Perform a training step
            loss = train_step(model, optimizer, images, angles)
            total_loss += loss
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


def evaluate_model(model, dataloader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles in dataloader:
            images = images.to(device)
            angles = angles.to(device)

            predictions = model(images)
            probs = F.softmax(predictions, dim=1)
            predicted_bins = torch.argmax(probs, dim=1)
        
            # Convert bins to angles and compare with ground truth
            predicted_angles = [bin_to_angle(bin_idx.item()) for bin_idx in predicted_bins]
            total += len(angles)
            correct += sum(1 for pred, true in zip(predicted_angles, angles) if abs(pred - true) < 15)  # Allow 15° margin

    print(f"Accuracy: {correct / total * 100:.2f}%")


def train( training_dataset_path, testing_dataset_path = None, model_weights_path = None):

    if not model_weights_path:
        model_weights_path = "./models/torch/yaw_model_weights.pth"

    # Initialize model and optimizer
    yaw_model = CarYawEstimator().to(device)
    optimizer = torch.optim.Adam(yaw_model.parameters(), lr=0.001)
    
    logging.info(f'Traning model with {training_dataset_path}')
    dataset = CarYawDataset(training_dataset_path, transform=yaw_model.transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_epochs_loop(yaw_model, optimizer, dataloader, num_epochs=10)

    if testing_dataset_path:
        logging.info(f'Evaluating model with {testing_dataset_path}')
        dataset = CarYawDataset(testing_dataset_path, transform=yaw_model.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    else:
        logging.info(f'Evaluating model with {training_dataset_path}')

    evaluate_model(yaw_model,dataloader)

    # Save the model weights after training
    logging.info(f'Saving model to {model_weights_path}')
    torch.save(yaw_model.state_dict(), model_weights_path)
    
    
if __name__ == "__main__":
    
    train()
    

