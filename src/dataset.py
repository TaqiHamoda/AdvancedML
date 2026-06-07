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
        # Input is in [0, 1]. Mean and std are based on dataset stats
        self.transform = v2.Compose([
            lambda x: torch.clamp(x, 0, 1),
            v2.Normalize(mean=[0.7706402555595372], std=[0.13980507597378528])
        ])

    def forward(self, img):
        return self.transform(img)


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian noise to the tensor to simulate sonar speckle/electronic noise."""
    def __init__(self, mean=0.0, sigma=(0.01, 0.05), p=0.5):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.p = p

    def forward(self, img):
        if torch.rand(1) >= self.p:
            return img

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        noise = torch.randn_like(img) * sigma + self.mean

        return img + noise


class TVGAttenuation(torch.nn.Module):
    """
    Simulates uncompensated propagation loss or TVG failure.
    Linearly decays pixel intensities from one side of the tile to the other.
    """
    def __init__(self, retention=(0.3, 0.7), p=0.5):
        """
        Args:
            retention (tuple): Range for the lowest multiplier at the faded edge.
                                   (e.g., 0.3 means the far edge loses up to 30% intensity).
            p (float): Probability of applying the augmentation.
        """
        super().__init__()

        if retention[0] > retention[1]:
            raise ValueError("retention[0] should be less than or equal to retention[1]")
        elif retention[0] < 0 or retention[1] > 1:
            raise ValueError("retention values should be in the range [0, 1]")

        self.retention = retention
        self.p = p

    def forward(self, img):
        if torch.rand(1) >= self.p:
            return img

        _, _, width = img.shape

        start = 1 - random.uniform(self.retention[0], self.retention[1])
        decay_line = torch.linspace(start, 1.0, steps=width, device=img.device, dtype=img.dtype)

        if random.random() < 0.5:  # Left-to-Right decay: Left side is intact (1.0), Right side is faded (drop_factor)
            decay_line = torch.flip(decay_line, dims=[0])  # Reverse to have decay from left to right

        return img * decay_line.view(1, 1, width)


class SonarDataset(Dataset):
    """
    Loads pre-processed .npy sonar tiles.
    Expected input: 384x384 numpy matrices, normalized [0, 1].
    Output: (1, 384, 384) FloatTensors.
    """
    def __init__(self, data_dir="data/processed", ext="*.npz"):
        super().__init__()

        self.files = sorted(glob.glob(os.path.join(data_dir, ext)))
        if len(self.files) == 0:
            raise ValueError(f"No files found in {data_dir} with extension {ext}")

        logger.info(f"Found {len(self.files)} sonar tiles.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        info = np.load(self.files[idx])
        data = info['data']
        distances = info['distances']

        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add channel dim -> (1, 384, 384)

        if distances.ndim == 2:
            distances = distances[np.newaxis, :, :]  # Add channel dim -> (1, 384, 384)

        data = torch.from_numpy(data)
        if torch.isnan(data).any() or torch.isinf(data).any():
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

        distances = torch.from_numpy(distances)
        if torch.isnan(distances).any() or torch.isinf(distances).any():
            distances = torch.nan_to_num(distances, nan=0.0, posinf=2.0, neginf=0.0)

        return data#, distances


class SonarDataTransform:
    """
    The DINOv3 data augmentation pipeline adapted for 1-channel Sonar data.
    Generates Global crops (for Teacher/Student) and Local crops (for Student).
    """
    def __init__(
        self,
        local_crops_number=8,
        global_crops_size=288,
        local_crops_size=96,
    ):
        self.local_crops_number = local_crops_number

        # Note: RandomResizedCrop is not used since it can distort the aspect ratio leading to uniform stretching
        # which breaks the physics of sonar imagery. Instead, we use RandomCrop to maintain the original aspect
        # ratio and spatial relationships.
        self.global_crop = v2.RandomCrop(global_crops_size)
        self.local_crop = v2.RandomCrop(local_crops_size)

        # Note: Blur shouldn't be used with sonar imagery since it breaks the physics of acoustics (no lens is used).
        # If you want to make the image or objects unclear, increase the noise instead to remain
        # accurate to the physics. (Interference is a more accurate way to model "blur" or loss of clarity)
        self.augmentations = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            TVGAttenuation(retention=(0.00, 0.75), p=0.3),  # TVG Attenuation to simulate propagation loss
            v2.RandomApply([v2.ColorJitter(
                brightness=(1.00, 1.15),                    # Brightness corresponds to sonar gain (intensity)
                contrast=(1.00, 2.00),                      # Contrast corresponds to dynamic range of reciever
                saturation=0, hue=0)                        # Data is only 1 channel and the concept of colors doesn't apply to sonar
            ], p=0.5),
            GaussianNoise(sigma=(0.00, 0.30), p=0.3),       # Gaussian Noise to simulate speckle noise
        ])

        self.normalize = NormalizeTransform()

    def __call__(self, image):
        teacher_crops = []
        student_crops = []

        # --- Global Crops (2 views) ---
        # Used by both Teacher and Student
        for _ in range(2):
            crop_aug = self.global_crop(image)
            teacher_crops.append(self.normalize(crop_aug.clone()))

            full_aug = self.augmentations(crop_aug)
            student_crops.append(self.normalize(full_aug))

        # --- Local Crops (8+ views) ---
        # Used by Student only to encourage local-to-global correspondence
        for _ in range(self.local_crops_number):
            crop_aug = self.local_crop(image)
            teacher_crops.append(self.normalize(crop_aug.clone()))

            full_aug = self.augmentations(crop_aug)
            student_crops.append(self.normalize(full_aug))

        return {
            'teacher': {
                'global_crops': teacher_crops[:2],
                'local_crops': teacher_crops[2:]
            },
            'student': {
                'global_crops': student_crops[:2],
                'local_crops': student_crops[2:]
            },
        }


class MaskingGenerator:
    def __init__(
        self,
        input_size=(288, 288),
        stride_size=32,
        mask_ratio=0.5,
        min_num_patches=4,     # Minimum size of a single block
        min_aspect=0.3,        # Min aspect ratio of block
    ):
        self.height, self.width = input_size[0] // stride_size, input_size[1] // stride_size
        self.num_patches = self.height * self.width

        self.min_num_patches = min_num_patches
        self.max_num_patches = int(mask_ratio * self.num_patches)

        self.log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def mask(self, mask, max_mask_patches):
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
        while mask_count < self.max_num_patches:
            max_mask_patches = self.max_num_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self.mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        # If we didn't reach the exact count (rare), fill the rest randomly
        if mask_count < self.max_num_patches:
             mask_flat = mask.flatten()
             to_add = np.random.choice(np.where(mask_flat == 0)[0], 
                                       size=self.max_num_patches - mask_count, 
                                       replace=False)
             mask_flat[to_add] = 1
             mask = mask_flat.reshape((self.height, self.width))

        return mask.flatten() # Returns (N_patches,) for compatibility with your training loop