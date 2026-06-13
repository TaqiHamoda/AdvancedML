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


class SparseDownsample(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride):
        super().__init__()
        self.norm = nn.LayerNorm(in_chans, eps=1e-6)
        self.conv = spconv.SparseConv2d(
            in_chans, out_chans, kernel_size=kernel_size, stride=stride, algo=spconv.ConvAlgo.Native
        )

    def forward(self, x: spconv.SparseConvTensor):
        # Apply norm to the flat active features (N, C)
        x = x.replace_feature(self.norm(x.features))
        return self.conv(x)


class SparseGRN(nn.Module):
    """Global Response Normalization computed natively on sparse spconv features."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, features, indices, batch_size):
        # features shape: (N_active, C)
        # indices shape: (N_active, 3) -> indices[:, 0] contains the batch index
        batch_idx = indices[:, 0].long()
        x2 = features.pow(2)
        sum_x2 = torch.zeros(batch_size, features.size(1), device=features.device, dtype=torch.float32)
        sum_x2.index_add_(dim=0, index=batch_idx, source=x2)
        Gx = torch.sqrt(sum_x2 + self.eps) # (B, C)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps) # (B, C)
        Nx_expanded = Nx[batch_idx] # (N_active, C)
        return self.gamma * (features * Nx_expanded) + self.beta + features


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = self.gamma * (x * Nx) + self.beta + x
        return out


class SparseDropPath(nn.Module):
    """DropPath adapted for sparse tensors. Drops entire samples in a batch, not individual tokens."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, features, indices, batch_size):
        if self.drop_prob == 0.0 or not self.training:
            return features

        keep_prob = 1 - self.drop_prob
        batch_idx = indices[:, 0].long()

        # Generate the random drop mask per batch element: shape (B, 1)
        random_tensor = keep_prob + torch.rand(batch_size, 1, dtype=features.dtype, device=features.device)
        random_tensor.floor_()  # 1 with keep_prob, 0 with drop_prob
        random_tensor.div_(keep_prob)

        drop_mask = random_tensor[batch_idx]
        return features * drop_mask


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


class SparseDepthwiseBypass(nn.Module):
    """
    Bypasses spconv's lack of depthwise support by utilizing PyTorch's native 
    highly-optimized dense depthwise convolutions, while perfectly preserving 
    the sparse Submanifold mathematical properties.
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim, 
            bias=True
        )
        
    def forward(self, x: spconv.SparseConvTensor):
        dense_x = x.dense()

        out = self.dwconv(dense_x)

        # x.indices is an (N, 3) tensor: [batch_idx, h_idx, w_idx]
        batch_idx = x.indices[:, 0].long()
        h_idx = x.indices[:, 1].long()
        w_idx = x.indices[:, 2].long()

        # Permute to (B, H, W, C) so we can efficiently index the active points
        out_hwc = out.permute(0, 2, 3, 1)
        active_features = out_hwc[batch_idx, h_idx, w_idx]

        # By only extracting active coordinates, we completely discard any kernel "bleed" 
        # into the empty space, strictly enforcing the Submanifold property.
        return x.replace_feature(active_features)


class SparseBlock(nn.Module):
    """Fully Sparse block for the Encoder."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # Sparse Depthwise Convolution
        # self.dwconv = spconv.SubMConv2d(
        #     in_channels=dim, out_channels=dim, kernel_size=7, padding=3, 
        #     groups=dim, bias=True, algo=spconv.ConvAlgo.Native
        # )
        self.dwconv = SparseDepthwiseBypass(dim=dim, kernel_size=7)

        self.norm = nn.LayerNorm(dim, eps=1e-6) 
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = SparseGRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)

        if drop_path > 0.:
            self.drop_path = SparseDropPath(drop_path)
        else:
            self.drop_path = lambda features, indices, batch_size: features  # Identity for sparse tensors

    def forward(self, x: spconv.SparseConvTensor):
        shortcut_features = x.features
        batch_size = x.batch_size

        x_sp = self.dwconv(x)
        features = x_sp.features
        indices = x_sp.indices

        features = self.norm(features)
        features = self.pwconv1(features)
        features = self.act(features)
        features = self.grn(features, indices, batch_size)
        features = self.pwconv2(features)
        features = self.gamma * features
        features = self.drop_path(features, indices, batch_size)

        return x_sp.replace_feature(shortcut_features + features)


class Block(nn.Module):
    """Dense block for the Decoder."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
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
        self.block = Block(dim=decoder_dim)
        self.head_proj = nn.Linear(decoder_dim, encoder_dim) 

    def forward(self, x, active_mask):
        x = self.proj(x)
        mask_expanded = active_mask.unsqueeze(1).type_as(x)
        
        # Inject the mask token into empty sites
        x = (x * mask_expanded) + (self.mask_token * (1.0 - mask_expanded))
        x = self.block(x)  # Mix spatial context
        
        # Flatten for the head
        return self.head_proj(x.flatten(2).transpose(1, 2))


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=1, drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Use Tiny ConvNeXtV2 configuration
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(SparseDownsample(in_chans, dims[0], kernel_size=4, stride=4))
        for i in range(3):
            self.downsample_layers.append(SparseDownsample(dims[i], dims[i+1], kernel_size=2, stride=2))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = nn.ModuleList([
                SparseBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) 
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]

        # Separate norms for 2D spatial patches vs 1D CLS token
        self.norm_patch = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.norm_cls = nn.LayerNorm(dims[-1], eps=1e-6) 

        self.decoder = ConvNeXtV2Decoder(encoder_dim=dims[-1], decoder_dim=512)

        self.apply(self._init_weights)
        self.embed_dim = dims[-1]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        if mask is not None:
            current_mask = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest').squeeze(1).bool()
        else:
            current_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=torch.bool)
        
        x_sparse = dense_to_sparse(x, current_mask)

        for i in range(4):
            x_sparse = self.downsample_layers[i](x_sparse)
            for block in self.stages[i]:
                x_sparse = block(x_sparse)

        x = x_sparse.dense()

        # Global CLS Branch
        if mask is not None:
            # Clean up bias leakage in empty space before Global Pooling
            mask_x = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest')
            x = x * mask_x

            # current_mask is True for active tokens. We count them to get the denominator.
            active_count = current_mask.sum(dim=(-1, -2)).unsqueeze(-1) + 1e-6
            # Sum over spatial dims and divide by valid tokens
            x_cls = x.sum([-2, -1]) / active_count
        else:
            x_cls = x.mean([-2, -1])

        x_cls = self.norm_cls(x_cls)
        x_patch_spatial = self.norm_patch(x) 

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
