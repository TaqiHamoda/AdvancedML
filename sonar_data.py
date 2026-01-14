import glob
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class GaussianNoise(torch.nn.Module):
    """Adds Gaussian noise to the tensor to simulate sonar speckle/electronic noise."""
    def __init__(self, mean=0.0, sigma=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.sigma + self.mean
            return img + noise
        return img

class SonarDataset(Dataset):
    """
    Loads pre-processed .npy sonar tiles.
    Expected input: 384x384 numpy matrices, normalized [0,1].
    Output: (1, 384, 384) FloatTensors.
    """
    def __init__(self, data_dir="dataset", ext="*.npy"):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, ext)))
        if len(self.files) == 0:
            raise ValueError(f"No files found in {data_dir} with extension {ext}")
        print(f"Found {len(self.files)} sonar tiles.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # Load npy file
        data = np.load(path).astype(np.float32)
        
        # Ensure it is (H, W) or (1, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, :, :] # Add channel dim -> (1, 384, 384)
            
        # Convert to tensor
        tensor = torch.from_numpy(data)
        
        # Safety check for NaNs/Infs which ruin contrastive learning
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            
        return tensor

class SonarDataTransform:
    """
    The DINOv3 data augmentation pipeline adapted for 1-channel Sonar data.
    Generates Global crops (for Teacher/Student) and Local crops (for Student).
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.local_crops_number = local_crops_number
        
        # 1. Geometric Augmentations (Spatial Invariance)
        # We use RandomResizedCrop to force the model to match features across scales.
        self.geo_global = v2.Compose([
            v2.RandomResizedCrop(global_crops_size, scale=global_crops_scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            # Optional: Vertical flip if your survey lines have varying headings
            v2.RandomVerticalFlip(p=0.5), 
        ])

        self.geo_local = v2.Compose([
            v2.RandomResizedCrop(local_crops_size, scale=local_crops_scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        # 2. Intensity Augmentations (Robustness to Gain/Contrast/Noise)
        # Note: We omit Hue/Saturation since we are 1-channel.
        self.intensity_trans = v2.Compose([
            v2.RandomApply([
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)
            ], p=0.8),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            # Custom Gaussian Noise for speckle robustness
            GaussianNoise(sigma=0.05, p=0.5), 
            # Normalize inputs (centering around 0 for neural net stability)
            # Assuming [0,1] input, (x - 0.5)/0.5 puts data in [-1, 1]
            v2.Normalize(mean=[0.5], std=[0.5]), 
        ])

    def __call__(self, image):
        # image is (1, 384, 384) tensor
        
        crops = []
        
        # --- Global Crops (2 views) ---
        # Used by both Teacher and Student
        for _ in range(2):
            geo_aug = self.geo_global(image)
            full_aug = self.intensity_trans(geo_aug)
            crops.append(full_aug)

        # --- Local Crops (8 views) ---
        # Used by Student only to encourage local-to-global correspondence
        for _ in range(self.local_crops_number):
            geo_aug = self.geo_local(image)
            full_aug = self.intensity_trans(geo_aug)
            crops.append(full_aug)

        return {
            'global_crops': crops[:2],
            'local_crops': crops[2:]
        }