from typing import Tuple, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from .dino import ConvNeXtV2, DINOHead
from .dataset import NormalizeTransform


def show_images(images: List[Tuple[np.ndarray, str]], num_images: int = 5, normalize: bool = True, cmap='gray'):
    transform = NormalizeTransform()

    _, axes = plt.subplots(1, num_images, figsize=(15, 3))
    if num_images == 1:
        axes = (axes, )

    for i, (image, title) in enumerate(images):
        img = transform(image).squeeze() if normalize else image
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.show()


def print_iou(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize row-wise (True labels) to handle large scale disparities
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    print("Confusion Matrix (Normalized):")

    header = f"{'True / Pred':>12} | " + " ".join([f"{str(label):>7}" for label in labels])
    print(header)
    print("-" * len(header))

    # Print each row with normalized values to 3 decimal places
    for i, row_label in enumerate(labels):
        row_str = " ".join([f"{val:>7.3f}" for val in cm_norm[i]])
        print(f"{str(row_label):>12} | {row_str}")

    print("-" * len(header))

    # --- Calculate and Print IoU on raw counts ---
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0
    )

    print("\nIoU per class:")
    for label, val in zip(labels, iou):
        print(f"  Class {str(label):<5}: {val:.3f}")

    print("-" * 20)
    print(f"Mean IoU (mIoU) : {np.mean(iou):.3f}")


def load_backbone(weights_path: str, device = torch.device("cuda")) -> ConvNeXtV2:
    """
    Loads the ConvNeXtTiny backbone from the training checkpoint.
    """
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    backbone_state_dict = {}
    if 'student' in checkpoint:
        backbone_state_dict = checkpoint['student']
    else:
        # Fallback if the user passes a raw state dict
        backbone_state_dict = checkpoint['backbone']

    if not backbone_state_dict:
        raise ValueError("No 'backbone.' keys found in the checkpoint. Check the weight file structure.")

    model = ConvNeXtV2(in_chans=1)
    model.load_state_dict(backbone_state_dict)
    model.to(device)
    model.eval()

    return model


def load_model(weights_path: str, output_dim: int = 4096, device = torch.device("cuda")) -> Tuple[ConvNeXtV2, DINOHead, DINOHead]:
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    dino_weights = checkpoint['student_dino_head']
    ibot_weights = checkpoint['student_ibot_head']

    backbone = load_backbone(weights_path, device)

    dino_head = DINOHead(in_dim=backbone.embed_dim, out_dim=output_dim)
    ibot_head = DINOHead(in_dim=backbone.embed_dim, out_dim=output_dim)

    for head, weights in ((dino_head, dino_weights), (ibot_head, ibot_weights)):
        head.load_state_dict(weights)
        head.to(device)
        head.eval()

    return backbone, dino_head, ibot_head


def run_inference(model: ConvNeXtV2, tile: np.ndarray, device = torch.device("cuda"), normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    transform = NormalizeTransform()

    with torch.no_grad():
        input_tensor = tile
        if normalize:
            input_tensor = transform(input_tensor)

        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Forward pass returns class embedding and patch embeddings
        stages = model._inference(input_tensor)

    outputs = []
    for stage in stages:
        outputs.append(stage.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    hypercolumn = model.fusion(stages)
    outputs.append(hypercolumn.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    cls, patch = model._get_output(hypercolumn)

    return cls.squeeze().cpu().detach().numpy(), patch.squeeze().cpu().detach().numpy(), outputs


def run_inference_heads(model: ConvNeXtV2, dino_head: DINOHead, ibot_head: DINOHead, tile: np.ndarray, device = torch.device("cuda")):
    cls, patch, outputs = run_inference(model, tile, device)

    dino_output = dino_head(torch.from_numpy(cls).to(device))
    ibot_output = ibot_head(torch.from_numpy(patch).to(device))

    return dino_output.squeeze().cpu().detach().numpy(), ibot_output.squeeze().cpu().detach().numpy(), outputs