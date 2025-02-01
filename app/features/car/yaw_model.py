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
        try:
            yaw_model.load_state_dict(torch.load(model_wts_path, map_location=device))
        except Exception as e:
            logging.error(f'🚨 setup_yaw_model error - {str(e)}')
            # continue without weights...

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
import shutil
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

# Consider circular distance instead of linear
def angle_distance(pred, true):
    diff = abs(pred - true)
    return min(diff, 360 - diff)


def compute_loss(predictions, target_bins):
    '''
        Custom loss function that handles circular nature of angles
        predictions: tensor of shape (batch_size, num_bins)
        targets: tensor of shape (batch_size,) containing bin indices
    '''
    
    # Consider adding weights to handle class imbalance
    weights = torch.ones(8).to(device)  # Modify based on class distribution
    ce_loss = F.cross_entropy(predictions, target_bins, weight=weights)

    # Standard cross-entropy loss
    # ce_loss = F.cross_entropy(predictions, target_bins)
    
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


def validate(model, dataloader):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    # criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # No gradients needed for validation
        for images, angles in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            
            # Convert target angles to bins
            target_bins = angle_to_bin_vectorized(angles)
    
            outputs = model(images)
            loss = compute_loss(outputs, target_bins)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            # Convert bins to angles and compare with ground truth
            probs = F.softmax(outputs, dim=1)
            predicted_bins = torch.argmax(probs, dim=1)
            predicted_angles = [bin_to_angle(bin_idx.item()) for bin_idx in predicted_bins]
            total += len(angles)

            correct += sum(1 for pred, true in zip(predicted_angles, angles) 
              if angle_distance(pred, true) < 15)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_one_epoch(model, optimizer, dataloader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    # criterion = nn.CrossEntropyLoss()
    
    for images, angles in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        
        target_bins = angle_to_bin_vectorized(angles)
   
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = compute_loss(outputs, target_bins)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate training accuracy
        # Convert bins to angles and compare with ground truth
        probs = F.softmax(outputs, dim=1)
        predicted_bins = torch.argmax(probs, dim=1)
        predicted_angles = [bin_to_angle(bin_idx.item()) for bin_idx in predicted_bins]
        total += len(angles)
        correct += sum(1 for pred, true in zip(predicted_angles, angles) 
              if angle_distance(pred, true) < 15)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_epochs_loop(model, optimizer, train_loader, val_loader, num_epochs=30, patience=5, min_delta=0.01,model_wts_path=None):
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 
                                                            mode = 'min', 
                                                            factor = 0.1, 
                                                            patience = 3, 
                                                            verbose = True    
                                                        )

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader)
        patience_msg = f', Patience: {patience_counter}' if patience_counter else ''

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%{patience_msg}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Accuracy: {val_acc:.2f}%")
        
        # With validation loss adjust optimizer step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if model_wts_path:
                torch.save(model.state_dict(), model_wts_path)
                print(f"  Saved model into {model_wts_path} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    return best_val_loss


def train(  training_dataset_path   = None, 
            testing_dataset_path    = None,
            model_weights_path      = None, 
            freeze_features_layers  = False,
            num_epochs              = 30
        ):

    if not model_weights_path:
        model_weights_path = "./models/torch/yaw_model_weights.pth"

    # Initialize model and optimizer
    yaw_model = CarYawEstimator().to(device)
    if model_weights_path:
        try:
            yaw_model.load_state_dict(torch.load(model_weights_path))
        except Exception as e:
            logging.error(f'🚨 train error - {str(e)}')
            #continue without weights...

    if training_dataset_path:
        logging.info(f'Training model with {training_dataset_path}')
        train_dataset = CarYawDataset(training_dataset_path, transform=yaw_model.transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    else:
        train_dataloader = None

    if testing_dataset_path:
        test_dataset = CarYawDataset(testing_dataset_path, transform=yaw_model.transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    else:
        test_dataloader = train_dataloader

    if not train_dataloader and not test_dataloader:
        logging.error(f'🚨 train : either training_dataset_path or testing_dataset_path are needed, not supplied')

    if not train_dataloader:
        logging.info(f'Evaluating model with {testing_dataset_path}')
        val_loss, val_acc = validate(yaw_model, test_dataloader)
        print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        return 

    # Freeze feature extractor layers?
    for param in yaw_model.backbone.features.parameters():
        param.requires_grad = freeze_features_layers  # Freeze feature layers

    if freeze_features_layers:
        optimizer = torch.optim.Adam(yaw_model.backbone.classifier.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.Adam(yaw_model.parameters(), lr=0.001)

    # epoch loop
    best_loss = train_epochs_loop(  yaw_model, 
                                    optimizer, 
                                    train_dataloader,
                                    test_dataloader, 
                                    num_epochs=num_epochs,
                                    model_wts_path=model_weights_path
                                    )    

    
if __name__ == "__main__":
    
    train()
    

