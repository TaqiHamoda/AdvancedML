from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch

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