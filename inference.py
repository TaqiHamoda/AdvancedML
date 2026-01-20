import os
import random
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_backbone(os.path.join(WEIGHTS_DIR, WEIGHT_FILE), device)

    print("Loading dataset...")
    try:
        dataset = SonarDataset()
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    total_files = len(dataset)
    if total_files < N_TILES:
        print(f"Warning: Requested {N_TILES} tiles but dataset only has {total_files}. Using all.")
        indices = list(range(total_files))
    else:
        indices = random.sample(range(total_files), N_TILES)

    transform = NormalizeTransform()

    print(f"Running inference on {len(indices)} tiles...")

    data = []
    with torch.no_grad():
        for idx in indices:
            raw_tensor = dataset[idx]
            input_tensor = transform(raw_tensor).unsqueeze(0).to(device)

            # Forward pass returns class embedding and patch embeddings
            cls, patch = model(input_tensor)
            data.append((
                raw_tensor.squeeze().cpu().numpy(),
                cls.squeeze().cpu().numpy(),
                patch.squeeze().cpu().numpy()
            ))

    # Concatenate all features from all images to learn a common color mapping
    print("Fitting PCA...")
    stacked_patches = np.concatenate([d[2] for d in data], axis=0) # (total patches, Dim)

    pca = PCA(n_components=3, whiten=True)
    pca.fit(stacked_patches)

    print("Saving visualizations...")

    patch_dim = int(np.sqrt(data[0][2].shape[0]))
    for i, (img, cls, patch) in enumerate(data):
        # Transform ALL pixels of this image
        pca_patch = pca.transform(patch) # (H8*W8, 3)

        # Sigmoid & Scaling
        pca_patch = 1.0 / (1.0 + np.exp(-2 * pca_patch))
        pca_patch = (255 * pca_patch).astype(np.uint8)
        pca_patch = pca_patch.reshape(patch_dim, patch_dim, 3)

        # Resize to original image size
        pca_vis = cv2.resize(pca_patch, img.shape, interpolation=cv2.INTER_NEAREST_EXACT)

        # Save Input
        raw_img = np.clip(img, 0, 1)
        gray_vis = (255 * raw_img).astype(np.uint8)

        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_org.png"), gray_vis)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_pca.png"), pca_vis)

    print(f"Done. Results saved to {RESULTS_DIR}")