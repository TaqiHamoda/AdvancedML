import os
import random
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA

# Import your modules
from dino import ConvNeXtTiny
from sonar_data import NormalizeTransform, SonarDataset

N_TILES = 100
RESULTS_DIR = "results/"
WEIGHTS_DIR = "weights/"
WEIGHT_FILE = "checkpoint_3.pth"


def load_backbone(weights_path, device):
    """
    Loads the ConvNeXtTiny backbone from the training checkpoint.
    """
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Extract student weights
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        # Fallback if the user passes a raw state dict
        state_dict = checkpoint

    # The training code wrapped the model in MultiCropWrapper, so keys have 'backbone.' prefix.
    # We need to strip this prefix to load into ConvNeXtTiny directly.
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            backbone_state_dict[new_key] = v
    
    if not backbone_state_dict:
        raise ValueError("No 'backbone.' keys found in the checkpoint. Check the weight file structure.")

    model = ConvNeXtTiny(in_chans=1)
    model.load_state_dict(backbone_state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_backbone(f"{WEIGHTS_DIR}/{WEIGHT_FILE}", device)

    print("Loading dataset...")
    try:
        dataset = SonarDataset()
    except ValueError as e:
        print(f"Error: {e}")
        return

    total_files = len(dataset)
    if total_files < N_TILES:
        print(f"Warning: Requested {N_TILES} tiles but dataset only has {total_files}. Using all.")
        indices = list(range(total_files))
    else:
        indices = random.sample(range(total_files), N_TILES)

    transform = NormalizeTransform()

    print(f"Running inference on {len(indices)} tiles...")

    all_features = []
    metadata = [] # Store original images and shapes for reconstruction
    with torch.no_grad():
        for idx in indices:
            # Load raw item (Tensor: 1, H, W)
            raw_tensor = dataset[idx]
            original_h, original_w = raw_tensor.shape[1], raw_tensor.shape[2]
            
            # Apply inference transform (Normalize)
            input_tensor = transform(raw_tensor).unsqueeze(0).to(device) # (1, 1, H, W)

            # Forward pass
            _, patch_tokens = model(input_tensor)
            features = patch_tokens[0].cpu().numpy() # (N_patches, Dim)
            all_features.append(features)
            metadata.append({
                'idx': idx,
                'original_tensor': raw_tensor, # Keep on CPU
                'grid_h': int(original_h / 32), # ConvNeXt stride is 32
                'grid_w': int(original_w / 32)
            })

    # Concatenate all features from all images to learn a common color mapping
    print("Fitting PCA...")
    stacked_features = np.concatenate(all_features, axis=0) # (Total_Patches, Dim)
    
    pca = PCA(n_components=3, whiten=True)
    pca_features = pca.fit_transform(stacked_features) # (Total_Patches, 3)

    # x * 2 -> sigmoid -> [0, 255]
    pca_features = 1.0 / (1.0 + np.exp(-2 * pca_features)) # Sigmoid for vibrant colors
    pca_features = (pca_features * 255).astype(np.uint8)

    print("Saving visualizations...")

    current_idx = 0
    for meta in metadata:
        n_patches = meta['grid_h'] * meta['grid_w']
        
        # Extract the specific features for this image
        img_pca_flat = pca_features[current_idx : current_idx + n_patches]
        current_idx += n_patches
        
        # Reshape to 2D grid
        pca_grid = img_pca_flat.reshape(meta['grid_h'], meta['grid_w'], 3)
        
        # Upsample PCA to original resolution
        original_h, original_w = meta['original_tensor'].shape[1], meta['original_tensor'].shape[2]
        pca_vis = cv2.resize(pca_grid, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        # Convert raw tensor to grayscale image [0, 255]
        raw_img = meta['original_tensor'].squeeze().numpy()
        # Ensure it's in [0, 1] before scaling (Dataset creates [0,1] floats)
        raw_img = np.clip(raw_img, 0, 1)
        grayscale_vis = (255 * raw_img).astype(np.uint8)

        # Save Grayscale Input
        gray_path = os.path.join(RESULTS_DIR, f"tile_{meta['idx']}_input.png")
        cv2.imwrite(gray_path, grayscale_vis)

        # Save PCA visualization
        pca_path = os.path.join(RESULTS_DIR, f"tile_{meta['idx']}_pca.png")
        cv2.imwrite(pca_path, pca_vis)

    print(f"Done. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()