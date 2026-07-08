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


class SpatialLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

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
    """Fully Sparse block."""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = SparseDepthwiseBypass(dim=dim, kernel_size=7)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = SparseGRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

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
        features = self.drop_path(features, indices, batch_size)

        return x_sp.replace_feature(shortcut_features + features)


class ConvNeXtV2Decoder(nn.Module):
    """Lightweight MAE decoder to provide context to masked patches."""
    def __init__(self, encoder_dim=768, decoder_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.block = SparseBlock(dim=decoder_dim)
        self.head_proj = nn.Linear(decoder_dim, encoder_dim)

    def forward(self, x, active_mask):
        x = self.proj(x)
        mask_expanded = active_mask.unsqueeze(1).type_as(x)

        # Inject the mask token into empty sites (re-densifies the tensor)
        x = (x * mask_expanded) + (self.mask_token * (1.0 - mask_expanded))

        # Create a 100% active mask to feed the dense grid into the SparseBlock
        full_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=torch.bool)
        x_sparse = dense_to_sparse(x, full_mask)

        x = self.block(x_sparse).dense()  # Mix spatial context

        # Flatten for the head
        return self.head_proj(x.flatten(2).transpose(1, 2))


class FeatureFusionBlock(nn.Module):
    """
    Fuses the output from multiple stages into a semantically compressed hypercolumn using an MLP.

    Sources: https://arxiv.org/abs/1411.5752
    """
    def __init__(self, stage_dims):
        super().__init__()

        n = len(stage_dims)
        if n < 2:
            raise ValueError("Number of stages to fuse has to be at least 2.")

        self.mlp = nn.Sequential()
        for i in range(n - 1):
            in_dim, out_dim = sum(stage_dims[i:]), sum(stage_dims[i + 1:])

            self.mlp.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            if i < n - 2:
                self.mlp.append(SpatialLayerNorm(out_dim))
                self.mlp.append(nn.GELU())

    def forward(self, stages):
        target_size = stages[0].shape[-2:]

        upsampled = [stages[0]]
        for s in stages[1:]:
            upsampled.append(F.interpolate(s, size=target_size, mode='bilinear', align_corners=False))

        fused = torch.cat(upsampled, dim=1)
        return self.mlp(fused)


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=1, drop_path_rate=0.0, arch="tiny"):
        super().__init__()

        if arch == "tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        elif arch == "base":
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]
        elif arch == "large":
            depths = [3, 3, 27, 3]
            dims = [192, 384, 768, 1536]
        elif arch == "huge":
            depths = [3, 3, 27, 3]
            dims = [352, 704, 1408, 2816]
        else:
            raise ValueError(f"Configuration {arch} doesn't exist. Please use one of the supported configurations: tiny, base, large, huge.")

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SpatialLayerNorm(dims[0])
        ))  # Add a dense stem first to prevent boundary artifacts from sparse conv
        for i in range(3):
            self.downsample_layers.append(SparseDownsample(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = nn.ModuleList([
                SparseBlock(dim=dims[i], drop_path=dp_rates[cur + j]) 
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]

        self.embed_dim = dims[-1]

        self.norm_patch = SpatialLayerNorm(self.embed_dim)
        self.norm_cls = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.fusion = FeatureFusionBlock(dims)
        self.decoder = ConvNeXtV2Decoder(encoder_dim=self.embed_dim, decoder_dim=dims[-2])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _inference(self, x, mask=None):
        """
        Runs the forward pass and returns the dense feature maps 
        from all 4 stages.
        """
        x = self.downsample_layers[0](x)

        if mask is not None:
            # Upsample mask precisely to match the spatial dimensions of the stem output.
            scale_factor = x.shape[-1] // mask.shape[-1]
            current_mask = mask.repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)

            mask_expanded = current_mask.unsqueeze(1)
            x = x * mask_expanded
        else:
            current_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=torch.bool)

        x_sparse = dense_to_sparse(x, current_mask)
        for block in self.stages[0]:
            x_sparse = block(x_sparse)

        outputs = [x_sparse.dense()]
        for i in range(1, 4):
            # Pass through the downsample layer and the blocks of the current stage
            x_sparse = self.downsample_layers[i](x_sparse)
            for block in self.stages[i]:
                x_sparse = block(x_sparse)

            # Convert back to dense for the intermediate output
            stage_out = x_sparse.dense()
            outputs.append(stage_out)

        return outputs

    def _get_output(self, x, mask=None):
        x_patch_spatial = self.norm_patch(x)

        if mask is not None:
            scale_h = x.shape[-2] // mask.shape[-2]
            scale_w = x.shape[-1] // mask.shape[-1]
            mask_x = mask.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)

            x_masked = x * mask_x.unsqueeze(1)
            active_count = mask_x.sum(dim=(-1, -2)).unsqueeze(-1) + 1e-6

            x_cls = x_masked.sum([-2, -1]) / active_count
            x_patch_flat = self.decoder(x_patch_spatial, mask_x)  # Reconstruct masked sites with context
        else:
            x_cls = x.mean([-2, -1])
            x_patch_flat = x_patch_spatial.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)

        x_cls = self.norm_cls(x_cls)

        return x_cls, x_patch_flat

    def _fcmae(self, x, mask=None):
        x = self._inference(x, mask)[-1]
        return self._get_output(x, mask)

    def forward(self, x, mask=None):
        x = self.fusion(self._inference(x, mask))
        return self._get_output(x, mask)


class ReconstructionHead(nn.Module):
    """
    A minimalist reconstruction head for masked image modeling pre-training.
    It maps the flattened patch representations back to pixel space using a 
    single strided ConvTranspose2d operation, keeping the heavy lifting in the backbone.
    """
    def __init__(self, embed_dim, patch_size=32, in_chans=1):
        super().__init__()
        self.head = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x_patch_flat, h_grid=None, w_grid=None):
        """
        x_patch_flat: (B, N, C) - Flattened patch features from ConvNeXtV2
        """
        B, N, C = x_patch_flat.shape

        # If grid dimensions aren't explicitly provided, assume a square image crop
        if h_grid is None or w_grid is None:
            h_grid = w_grid = int(N ** 0.5)

        # (B, N, C) -> (B, C, N) -> (B, C, h_grid, w_grid)
        x_spatial = x_patch_flat.transpose(1, 2).view(B, C, h_grid, w_grid)

        # (B, C, h_grid, w_grid) -> (B, in_chans, h_grid*32, w_grid*32)
        reconstructed_pixels = self.head(x_spatial)

        return reconstructed_pixels


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, bottleneck_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
