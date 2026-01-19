import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise conv: groups=dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)

        # Pointwise convs implemented as Linear (1x1 conv)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=1, drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # ConvNeXt Tiny Config
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        # Stem: (N, 1, H, W) -> (N, 96, H/4, W/4)
        self.downsample_layers = nn.ModuleList() 
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))

        # Downsampling between stages
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            ))
            cur += depths[i]

        # Norms for Fusion input features (Standardizing before concat)
        self.norm_stages = nn.ModuleList()
        for i in range(2, 4):
            self.norm_stages.append(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            )

        # Projection: (384 + 768 = 1152) -> 768.
        self.fusion_proj = nn.Linear(dims[2] + dims[3], dims[-1])

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        self.apply(self._init_weights)
        self.embed_dim = dims[-1] # 768

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (N, 1, H, W)
        hypercolumn = None
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i < 2:
                continue

            x_i = self.norm_stages[i - 2](x)
            if hypercolumn is None:
                hypercolumn = x_i
            else:
                x_i = F.interpolate(x_i, size=hypercolumn.shape[-2:], mode='bilinear')
                hypercolumn = torch.cat([hypercolumn, x_i], dim=1)

        # --- Global CLS Branch ---
        # x is now (N, 768, H/32, W/32)
        # Global Pooling for [CLS] token analog
        x_cls = x.mean([-2, -1]) # (N, 768)
        x_cls = self.norm(x_cls)

        # --- Patch Fusion Branch ---
        # Project & Final Normalize
        x_patch = hypercolumn.flatten(2).transpose(1, 2)   # (N, 196, 1152)
        x_patch = self.fusion_proj(x_patch)                # (N, 196, 768)
        x_patch = self.norm(x_patch)                # Normalized (N, 196, 768)

        return x_cls, x_patch


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # The last layer (prototypes) requires Weight Normalization in DINO
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1)  # original0 is magnitude. original1 is direction
            if norm_last_layer:
                self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # L2 normalize bottleneck
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Standard DINO wrapper. 
    Forward pass handles a list of crops with different resolutions.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Case 1: Input is a single tensor (e.g., validation or standard forward)
        if not isinstance(x, list):
            cls_token, patch_tokens = self.backbone(x)
            return self.head(cls_token), [patch_tokens], cls_token

        # Case 2: Input is a list of crops (training)
        output_cls_tokens = []
        output_patch_tokens_list = []

        start_idx = 0
        n_crops = len(x)
        current_res = x[0].shape[-1]

        # Iterate to find boundaries where resolution changes
        for i in range(1, n_crops + 1):
            # If we reached the end or the resolution changed
            if i == n_crops or x[i].shape[-1] != current_res:
                end_idx = i

                # Concatenate the contiguous block of crops
                # shape: (Batch * n_crops_in_block, C, H, W)
                block_input = torch.cat(x[start_idx:end_idx])

                # Forward pass
                _out_cls, _out_patch = self.backbone(block_input)
                output_cls_tokens.append(_out_cls)
                output_patch_tokens_list.append(_out_patch)

                # Prepare for next block
                if i < n_crops:
                    start_idx = i
                    current_res = x[i].shape[-1]

        output_cls = torch.cat(output_cls_tokens)

        # Keep patch tokens separate (dimensions differ: 224->(7x7) patches, 96->(3x3) patches)
        return self.head(output_cls), output_patch_tokens_list, output_cls
