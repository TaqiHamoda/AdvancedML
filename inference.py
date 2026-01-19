import os
import random
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA
import torch.nn.functional as F

# Import your modules
from dino import ConvNeXtTiny
from sonar_data import NormalizeTransform, SonarDataset

RESULTS_DIR = "results/"
WEIGHTS_DIR = "weights/"
WEIGHT_FILE = "checkpoint_latest.pth"

N_TILES = 100
FEATURE_WEIGHTS = (0.1, 0.1, 1)


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

    model = load_backbone(os.path.join(WEIGHTS_DIR, WEIGHT_FILE), device)

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
            raw_tensor = dataset[idx]
            input_tensor = transform(raw_tensor).unsqueeze(0).to(device)

            # Forward pass returns list of 3 tensors
            feats_list = model.get_features_stages(input_tensor)
            
            # feats_list[0] is the largest (Target Resolution: H/8, W/8)
            target_h, target_w = feats_list[0].shape[-2:]

            resized_feats = []

            # Process Stage 1 (already target size)
            f1 = feats_list[0] # (1, 192, H8, W8)
            resized_feats.append(FEATURE_WEIGHTS[0] * f1)

            # Process Stage 2 (Upsample 2x)
            f2 = FEATURE_WEIGHTS[1] * F.interpolate(feats_list[1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_feats.append(f2)

            # Process Stage 3 (Upsample 4x)
            f3 = FEATURE_WEIGHTS[2] * F.interpolate(feats_list[2], size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_feats.append(f3)

            # Concatenate along channel dimension
            # 192 + 384 + 768 = 1344 dimensions
            hypercolumn = torch.cat(resized_feats, dim=1) # (1, 1344, H8, W8)

            # Flatten to (N_pixels, 1344)
            features_flat = F.normalize(hypercolumn.permute(0, 2, 3, 1).flatten(0, 2), dim=1).cpu().numpy()

            all_features.append(features_flat)
            metadata.append({
                'idx': idx,
                'original_tensor': raw_tensor,
                'feat_h': target_h,
                'feat_w': target_w
            })

    # Concatenate all features from all images to learn a common color mapping
    print("Fitting PCA...")
    stacked_features = np.concatenate(all_features, axis=0) # (Total_Patches, Dim)

    pca = PCA(n_components=3, whiten=True)
    pca.fit(stacked_features) # (Total_Patches, 3)

    print("Saving visualizations...")

    for i, meta in enumerate(metadata):
        # Transform ALL pixels of this image
        feats = all_features[i]
        pca_feats = pca.transform(feats) # (H8*W8, 3)

        # Sigmoid & Scaling
        pca_feats = 1.0 / (1.0 + np.exp(-2 * pca_feats))
        pca_feats = (255 * pca_feats).astype(np.uint8)

        # Reshape to grid
        h, w = meta['feat_h'], meta['feat_w']
        pca_grid = pca_feats.reshape(h, w, 3)

        # Resize to original image size
        orig_h, orig_w = meta['original_tensor'].shape[1], meta['original_tensor'].shape[2]
        pca_vis = cv2.resize(pca_grid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST_EXACT)

        # Save Input
        raw_img = meta['original_tensor'].squeeze().numpy()
        raw_img = np.clip(raw_img, 0, 1)
        gray_vis = (255 * raw_img).astype(np.uint8)

        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_org.png"), gray_vis)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_pca.png"), pca_vis)

    print(f"Done. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()