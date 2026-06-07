import os
import random
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import your modules
from src.dino import ConvNeXtTiny
from src.dataset import NormalizeTransform, SonarDataset

LOG_FILE = "logs/1769267466.450234.log"

FIGURES_DIR = "figures/"
RESULTS_DIR = "results/"
WEIGHTS_DIR = "weights/"
WEIGHT_FILE = "checkpoint_latest.pth"

N_TILES = 100


def save_figure(graphs, labels, name, threshold=None):
    plt.figure(figsize=(12, 6))

    # Convert lists to numpy arrays for easier plotting
    iters = range(len(graphs[0]))

    # Draw a dotted red line at threshold
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle=':')

    # Plot DINO and Ibot
    for i in range(len(graphs)):
        plt.plot(iters, graphs[i], label=labels[i])

    # Add labels and title
    plt.title(name)
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

    # Save the plot to a file
    plt.savefig(f'{FIGURES_DIR}/{name}.png', format='png', dpi=300)  # You can change the filename and format as needed
    plt.close()


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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Creating figures...")

    data_log = {
        "lr": [],
        "temp": [],
        "m": [],
        "wd": [],
        "dino": [],
        "ibot": [],
        "gram": [],
        "koleo": []
    }

    lines = None
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "Epoch" not in line:
            continue

        l = line.split("lr: ")[-1]
        l = l.split(',')

        lr = float(l[0])
        data_log["lr"].append(lr)

        temp = float(l[1].split(': ')[-1])
        data_log["temp"].append(temp)

        m = float(l[2].split(': ')[-1])
        data_log["m"].append(m)

        wd = float(l[3].split(': ')[-1])
        data_log["wd"].append(wd)

        dino = float(l[4].split(': ')[-1])
        data_log["dino"].append(dino)

        ibot = float(l[5].split(': ')[-1])
        data_log["ibot"].append(ibot)

        gram = float(l[6].split(': ')[-1])
        data_log["gram"].append(gram)

        koleo = float(l[7].split(': ')[-1])
        data_log["koleo"].append(koleo)

    save_figure((data_log['dino'], data_log['ibot']), ('DINO', 'iBOT'), "dino_ibot_loss", np.log(4096))
    save_figure((data_log['gram'],), ('Gram',), "gram_loss")
    save_figure((data_log['koleo'],), ('KoLeo',), "koleo_loss")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    patch_dim = dataset[0].shape[-1]

    data = []
    with torch.no_grad():
        for idx in indices:
            raw_tensor = dataset[idx]
            input_tensor = transform(raw_tensor).unsqueeze(0).to(device)

            # Forward pass returns class embedding and patch embeddings
            cls, patch = model(input_tensor)

            # Upsample in the feature space (done in cpu so can be done while training)
            downsample_dim = int(np.sqrt(patch.shape[1]))
            upsampled_patch = patch.reshape((patch.shape[0], downsample_dim, downsample_dim, patch.shape[2]))
            upsampled_patch = upsampled_patch.permute(0, 3, 1, 2)
            upsampled_patch = F.interpolate(upsampled_patch.cpu(), size=(patch_dim, patch_dim), mode='bicubic', align_corners=False)
            upsampled_patch = upsampled_patch.permute(0, 2, 3, 1)

            data.append((
                raw_tensor.squeeze().cpu().numpy(),
                cls.squeeze().cpu().numpy(),
                patch.squeeze().cpu().numpy(),
                upsampled_patch.squeeze().cpu().numpy()
            ))

    # Concatenate all features from all images to learn a common color mapping
    print("Fitting PCA...")
    stacked_patches = np.concatenate([d[2] for d in data], axis=0) # (total patches, Dim)

    pca = PCA(n_components=3, whiten=True)
    pca.fit(stacked_patches)

    print("Saving visualizations...")

    for i, (img, cls, patch, upsampled_patch) in enumerate(data):
        # Visualize Sonar Image
        gray_vis = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        gray_vis = cv2.cvtColor(gray_vis, cv2.COLOR_GRAY2BGR)

        # Transform ALL features of this image to pixel space
        pca_patch = upsampled_patch.reshape(-1, upsampled_patch.shape[-1])
        pca_patch = pca.transform(pca_patch) # (H*W, 3)
        pca_patch = pca_patch.reshape(patch_dim, patch_dim, 3)

        # Sigmoid & Scaling for more vibrant colors
        pca_patch = 0.7 * pca_patch + 0.3 * img.reshape((patch_dim, patch_dim, 1))
        pca_patch = 1.0 / (1.0 + np.exp(-2 * pca_patch))
        pca_patch = (255 * pca_patch).astype(np.uint8)

        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_org.png"), gray_vis)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_pca.png"), pca_patch)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{i}_joined.png"), np.hstack((gray_vis, pca_patch)))

    print(f"Done. Results saved to {RESULTS_DIR}")