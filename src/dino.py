import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


def dense_to_sparse(x, mask):
    """Converts a dense tensor to a spconv SparseConvTensor."""
    B, C, H, W = x.shape
    indices = mask.nonzero(as_tuple=False).contiguous().int()
    x_hwc = x.permute(0, 2, 3, 1).contiguous()
    features = x_hwc[mask]
    return spconv.SparseConvTensor(features, indices, [H, W], B)


class DropPath(nn.Module):
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


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x, mask=None):
        if mask is not None:  # Prevent biases from leaking into padded areas
            mask_hwc = mask.unsqueeze(-1)
            x = x * mask_hwc

        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        if mask is not None:  # Average only over the active sites if mask is provided
            active_count = mask.sum(dim=(1,2), keepdim=True).unsqueeze(-1) + 1e-6
            Nx = Gx / (Gx.sum(dim=-1, keepdim=True) / active_count + 1e-6)
        else:
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        out = self.gamma * (x * Nx) + self.beta + x
        if mask is not None:  # Clean up bias leakage again
            out = out * mask_hwc

        return out


class Block(nn.Module):
    """Sparse block for the Encoder."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = spconv.SubMConv2d(
            in_channels=dim, out_channels=dim, kernel_size=7, padding=3, 
            groups=dim, bias=True, algo=spconv.ConvAlgo.Native
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: spconv.SparseConvTensor, mask: torch.Tensor):
        shortcut = x.dense()
        x_sp = self.dwconv(x)
        x_dense = x_sp.dense().permute(0, 2, 3, 1)
        x_dense = self.norm(x_dense)
        x_dense = self.pwconv1(x_dense)
        x_dense = self.act(x_dense)
        x_dense = self.grn(x_dense, mask) 
        x_dense = self.pwconv2(x_dense)
        if self.gamma is not None:
            x_dense = self.gamma * x_dense
        x_dense = x_dense.permute(0, 3, 1, 2) * mask.unsqueeze(1)
        out_dense = shortcut + self.drop_path(x_dense)
        return dense_to_sparse(out_dense, mask)


class DenseBlock(nn.Module):
    """Dense block for the Decoder."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + self.drop_path(x)


class ConvNeXtV2Decoder(nn.Module):
    """Lightweight MAE decoder to provide context to masked patches."""
    def __init__(self, encoder_dim=768, decoder_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.block = DenseBlock(dim=decoder_dim)
        self.head_proj = nn.Linear(decoder_dim, encoder_dim) 

    def forward(self, x, active_mask):
        x = self.proj(x)
        mask_expanded = active_mask.unsqueeze(1).type_as(x)
        
        # Inject the mask token into empty sites
        x = (x * mask_expanded) + (self.mask_token * (1.0 - mask_expanded))
        x = self.block(x)  # Mix spatial context
        
        # Flatten for the head
        return self.head_proj(x.flatten(2).transpose(1, 2))


class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=1, drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList() 
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = nn.ModuleList([
                Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) 
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]

        self.norm_stages = nn.ModuleList()
        for i in range(1, 4):
            self.norm_stages.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))

        # Projection is now spatial (Conv2d) instead of Linear
        self.fusion_proj = nn.Conv2d(dims[1] + dims[2] + dims[3], dims[-1], kernel_size=1)
        
        # Separate norms for 2D spatial patches vs 1D CLS token
        self.norm_patch = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.norm_cls = nn.LayerNorm(dims[-1], eps=1e-6) 
        
        # Embedded Decoder
        self.decoder = ConvNeXtV2Decoder(encoder_dim=dims[-1], decoder_dim=512)

        self.apply(self._init_weights)
        self.embed_dim = dims[-1]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        hypercolumn = None  
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            if mask is not None:
                current_mask = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest').squeeze(1).bool()
                x_sparse = dense_to_sparse(x, current_mask)
            else:
                current_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=torch.bool)
                x_sparse = dense_to_sparse(x, current_mask)

            for block in self.stages[i]:
                x_sparse = block(x_sparse, current_mask)

            x = x_sparse.dense()
            if i < 1: continue

            x_i = self.norm_stages[i - 1](x)
            if mask is not None:  # Clean up bias leakage before Hypercolumn
                mask_i = F.interpolate(mask.unsqueeze(1).float(), size=x_i.shape[-2:], mode='nearest')
                x_i = x_i * mask_i

            if hypercolumn is None:
                hypercolumn = x_i
            else:
                x_i = F.interpolate(x_i, size=hypercolumn.shape[-2:], mode='bilinear')
                hypercolumn = torch.cat([hypercolumn, x_i], dim=1)

        # Global CLS Branch
        if mask is not None:
            # Clean up bias leakage before Global Pooling
            mask_x = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest')
            x = x * mask_x

            # current_mask is True for active tokens. We count them to get the denominator.
            active_count = current_mask.sum(dim=(-1, -2), keepdim=True) + 1e-6
            # Sum over spatial dims and divide by valid tokens
            x_cls = x.sum([-2, -1]) / active_count
        else:
            x_cls = x.mean([-2, -1])

        x_cls = self.norm_cls(x_cls)

        # Patch Fusion Branch (Spatial)
        x_patch_spatial = self.fusion_proj(hypercolumn)
        x_patch_spatial = self.norm_patch(x_patch_spatial) 

        # Conditional Decoding
        if mask is not None:
            # Reconstruct masked sites with context
            x_patch_flat = self.decoder(x_patch_spatial, mask)
        else:
            # Standard dense flattening (B, C, H, W) -> (B, H*W, C)
            x_patch_flat = x_patch_spatial.flatten(2).transpose(1, 2)

        return x_cls, x_patch_flat


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

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # L2 normalize bottleneck

        if self.norm_last_layer:
            # Manually apply weight normalization (L2 norm of weight vector = 1)
            w = F.normalize(self.last_layer.weight, p=2, dim=1)
            x = F.linear(x, w)
        else:
            x = self.last_layer(x)

        return x


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, masks=None):
        if not isinstance(x, list):
            return self.backbone(x, mask=masks)

        output_cls_tokens = []
        output_patch_tokens_list = []

        start_idx = 0
        n_crops = len(x)
        current_res = x[0].shape[-1]
        
        # If no masks are passed, create a list of Nones
        if masks is None:
            masks = [None] * n_crops

        for i in range(1, n_crops + 1):
            if i == n_crops or x[i].shape[-1] != current_res:
                end_idx = i
                block_input = torch.cat(x[start_idx:end_idx])
                
                # Check if this crop block has masks
                block_masks = masks[start_idx:end_idx]
                if all(m is None for m in block_masks):
                    block_mask = None
                else:
                    block_mask = torch.cat(block_masks)

                # Pass both images and masks to ConvNeXt
                _out_cls, _out_patch = self.backbone(block_input, mask=block_mask)
                output_cls_tokens.append(_out_cls)
                output_patch_tokens_list.append(_out_patch)

                if i < n_crops:
                    start_idx = i
                    current_res = x[i].shape[-1]

        output_cls = torch.cat(output_cls_tokens)
        return self.head(output_cls), output_patch_tokens_list, output_cls
