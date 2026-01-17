import numpy as np
import os, glob, logging, random, math

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


class NormalizeTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Normalize inputs (centering around 0 for neural net stability)
        # Input is in [0, 1], (x - 0.5)/0.5 puts data in [-1, 1]
        self.transform = v2.Compose([
            lambda x: torch.clamp(x, 0, 1),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])

    def forward(self, img):
        return self.transform(img)


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


class MaskingGenerator:
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=32,
        mask_ratio=0.5,
        min_num_patches=4,     # Minimum size of a single block
        max_num_patches=None,  # Max size of a single block (defaults to varying)
        min_aspect=0.3,        # Min aspect ratio of block
        max_aspect=None,       # Max aspect ratio of block
    ):
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = int(mask_ratio * self.num_patches)

        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches if max_num_patches else self.num_masking_patches
        
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        """
        Tries to mask a random block with random aspect ratio.
        """
        delta = 0
        for _ in range(10): # Try 10 times to find a valid block
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                
                # Check if we are overlapping too much or adding too many patches
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=(self.height, self.width), dtype=int)
        mask_count = 0

        # Repeatedly add blocks until we reach the target mask ratio
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        # If we didn't reach the exact count (rare), fill the rest randomly
        if mask_count < self.num_masking_patches:
             mask_flat = mask.flatten()
             to_add = np.random.choice(np.where(mask_flat == 0)[0], 
                                       size=self.num_masking_patches - mask_count, 
                                       replace=False)
             mask_flat[to_add] = 1
             mask = mask_flat.reshape((self.height, self.width))

        return mask.flatten() # Returns (N_patches,) for compatibility with your training loop


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
        logger.info(f"Found {len(self.files)} sonar tiles.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx]).astype(np.float32)

        if data.ndim == 2:
            data = data[np.newaxis, :, :] # Add channel dim -> (1, 384, 384)

        tensor = torch.from_numpy(data)
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

        # We use RandomResizedCrop to force the model to match features across scales.
        self.geo_global = v2.Compose([
            v2.RandomResizedCrop(global_crops_size, scale=global_crops_scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5), 
        ])

        self.geo_local = v2.Compose([
            v2.RandomResizedCrop(local_crops_size, scale=local_crops_scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        # Brightness corresponds to sonar gain (intensity)
        # Contrast corresponds to dynamic range of reciever
        self.intensity_trans = v2.Compose([
            # v2.RandomApply([
            #     v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)
            # ], p=0.8),
            # v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            # GaussianNoise(sigma=0.05, p=0.5),  # Gaussian Noise to simulate speckle noise
            NormalizeTransform(),
        ])

    def __call__(self, image):
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