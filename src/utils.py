import numpy as np
import torch

from .dino import ConvNeXtV2
from .dataset import NormalizeTransform


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


def run_inference(model: ConvNeXtV2, tile: np.ndarray, device = torch.device("cuda")):
    transform = NormalizeTransform()

    with torch.no_grad():
        input_tensor = transform(tile).unsqueeze(0).to(device)

        # Forward pass returns class embedding and patch embeddings
        stages = model._inference(input_tensor)

    outputs = []
    for stage in stages:
        outputs.append(stage.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    hypercolumn = model.fusion(stages)
    outputs.append(hypercolumn.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy())

    cls, _ = model._get_output(hypercolumn)

    return cls.squeeze().cpu().detach().numpy(), outputs