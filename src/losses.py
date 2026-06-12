import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SinkhornKnopp(nn.Module):
    """
    DINOv2/3-style Loss: SinkhornKnopp + Softmax. 
    Better for large batch sizes (B >> K).
    """
    def __init__(self, out_dim, student_temp=0.1, n_iterations=3, eps=1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.n_iterations = n_iterations
        self.eps = eps

    @torch.no_grad()
    def get_probs(self, student_output, teacher_output, teacher_temp):
        # Cast to float32 to prevent overflow
        student_output = student_output.float()
        teacher_output = teacher_output.float()

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Max normalization for stability
        log_Q = (teacher_output / teacher_temp).t()  # (K, B*world_size)
        log_Q = log_Q - log_Q.max(dim=0, keepdim=True)[0]

        Q = torch.exp(log_Q)
        B = Q.shape[1] * world_size 
        K = Q.shape[0] 

        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q + self.eps

        for _ in range(self.n_iterations):
            row_sum = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(row_sum)
            Q /= row_sum + self.eps
            Q /= K

            col_sum = torch.sum(Q, dim=0, keepdim=True)
            Q /= col_sum + self.eps
            Q /= B

        Q *= B
        t_probs = Q.t()

        # Student Log-Softmax
        s_out = student_output / self.student_temp
        s_log_probs = F.log_softmax(s_out, dim=-1)

        return s_log_probs, t_probs

class Centering(nn.Module):
    """
    DINOv1-style Loss: Centering + Softmax. 
    More stable for small batch sizes (B < K).
    """
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.996):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center with Exponential Moving Average (EMA).
        """
        if torch.isnan(teacher_output).any():
            return  # Skip update if NaN detected

        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # Update in-place using copy_ (The assignment operator removes the registered buffer reference)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))

    def get_probs(self, student_output, teacher_output, teacher_temp):
        self.update_center(teacher_output)

        # Cast to float32 to prevent overflow
        student_output = student_output.float()
        teacher_output = teacher_output.float()

        # Center the Teacher Logits
        # teacher_output is (T*B, D)
        teacher_out_centered = teacher_output - self.center

        # Apply Softmax to Teacher (max normalization for stability)
        teacher_logits = teacher_out_centered / teacher_temp
        teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True)[0]
        t_probs = F.softmax(teacher_logits, dim=-1).detach()

        # Student Log-Softmax
        s_out = student_output / self.student_temp
        s_log_probs = F.log_softmax(s_out, dim=-1)

        return s_log_probs, t_probs


class DINOLoss(Centering):
    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Args:
            student_output: (S * B, D) student logits
            teacher_output: (T * B, D) teacher logits
            teacher_temp: scalar temperature
        """
        s_log_probs, t_probs = self.get_probs(student_output, teacher_output, teacher_temp)

        n_teacher_crops = 2
        B = teacher_output.shape[0] // n_teacher_crops

        # Reshape to (n_crops, B, D)
        n_student_crops = student_output.shape[0] // B
        t_probs = t_probs.view(n_teacher_crops, B, -1)
        s_log_probs = s_log_probs.view(n_student_crops, B, -1)

        # Cross-Entropy Loss
        loss_matrix = -torch.einsum("sbk,tbk->st", s_log_probs, t_probs)

        # Remove diagonal (same crop comparison)
        min_crops = min(n_student_crops, n_teacher_crops)
        loss_matrix = torch.diagonal_scatter(loss_matrix, torch.zeros(min_crops, device=loss_matrix.device))

        total_loss = loss_matrix.sum()
        normalization = B * (n_student_crops * n_teacher_crops - min_crops)

        return total_loss / normalization


class iBOTPatchLoss(Centering):
    def forward(self, student_patches, teacher_patches, masks, teacher_temp):
        s_log_probs, t_probs = self.get_probs(student_patches, teacher_patches, teacher_temp)

        # Cross Entropy
        loss_per_token = torch.sum(-t_probs * s_log_probs, dim=-1)

        # Weighted Mean
        n_masked_per_image = masks.sum(dim=1).clamp(min=1.0)
        weights_per_image = 1.0 / n_masked_per_image
        weights_expanded = weights_per_image.unsqueeze(1).expand_as(masks)
        weights_flat = weights_expanded[masks.bool()]

        return (loss_per_token * weights_flat).sum() / masks.shape[0]


class GramLoss(nn.Module):
    """
    Matches the texture/autocorrelation of features between Student and Teacher.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_patches, teacher_patches):
        student_patches = F.normalize(student_patches, dim=-1)
        teacher_patches = F.normalize(teacher_patches, dim=-1)

        student_gram = torch.bmm(student_patches.transpose(1, 2), student_patches)
        teacher_gram = torch.bmm(teacher_patches.transpose(1, 2), teacher_patches)

        return self.mse(student_gram, teacher_gram)


class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps  # Better to use a high epsilon (like 1e-4) to prevent gradient explosion
        self.pdist = nn.PairwiseDistance(2, eps=eps)

    def forward(self, student_output):
        x = F.normalize(student_output, dim=-1, p=2)

        # Gather all features from other GPUs
        with torch.no_grad():
            if dist.is_initialized():
                # Gather all tensors
                gathered_x = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_x, x)
                all_x = torch.cat(gathered_x, dim=0)

                # Identify which part of the global batch is local
                # We compute gradients only for local x, but use all_x for neighbor search
                rank = dist.get_rank()
            else:
                all_x = x

        # Compute dot products between LOCAL features and GLOBAL features
        # shape: (Local_B, Global_B)
        dots = torch.mm(x, all_x.t()) 

        # Mask the diagonal (self-matches)
        # The diagonal for the local batch corresponds to indices [rank*B : (rank+1)*B]
        if dist.is_initialized():
            rank = dist.get_rank()
            start_idx = rank * x.shape[0]
            for i in range(x.shape[0]):  # Mask the specific self-correlations
                dots[i, start_idx + i] = -1.0
        else:
            dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)

        # Find nearest neighbor in the global batch
        _, indices = torch.max(dots, dim=1)
        nearest_neighbors = all_x[indices]
        distances = self.pdist(x, nearest_neighbors)

        return -torch.log(distances).mean()


class HSICLoss(nn.Module):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) to measure 
    dependence between learned features and a target variable (e.g., distance).
    Minimizing this loss encourages the features to be independent of the target.

    Reference: https://arxiv.org/abs/2106.08320
    """
    def __init__(self):
        super().__init__()

    def rbf_kernel(self, X):
        # X shape: [N, D]
        X_sq = torch.sum(X ** 2, dim=-1)
        # Compute squared Euclidean distances: ||x_i - x_j||^2
        dist_sq = X_sq.unsqueeze(1) + X_sq.unsqueeze(0) - 2 * torch.matmul(X, X.t())
        dist_sq = torch.clamp(dist_sq, min=0.0) # Numerical stability

        # Median heuristic for sigma
        upper_tri = dist_sq.triu(diagonal=1).flatten()
        upper_tri = upper_tri[upper_tri > 0]

        if upper_tri.numel() > 0:
            median_dist_sq = torch.median(upper_tri)
        else:
            median_dist_sq = torch.tensor(1.0, device=dist_sq.device)

        sigma_sq = torch.clamp(median_dist_sq, min=1e-4)

        # RBF Kernel K(x_i, x_j) = exp(-||x_i - x_j||^2 / 2*sigma^2)
        return torch.exp(-dist_sq / (2 * sigma_sq))

    def forward(self, features, targets):
        """
        features: [N, D_f] (Patch features flattened across batch)
        targets:  [N, D_t] (Patch distances flattened across batch)
        """
        N = features.size(0)

        K = self.rbf_kernel(features)
        L = self.rbf_kernel(targets)

        # Double centering the matrices
        K_mean_row = K.mean(dim=0, keepdim=True)
        K_mean_col = K.mean(dim=1, keepdim=True)
        K_mean_all = K.mean()
        Kc = K - K_mean_row - K_mean_col + K_mean_all

        L_mean_row = L.mean(dim=0, keepdim=True)
        L_mean_col = L.mean(dim=1, keepdim=True)
        L_mean_all = L.mean()
        Lc = L - L_mean_row - L_mean_col + L_mean_all

        # HSIC calculation (Trace of Kc * Lc normalized)
        return torch.sum(Kc * Lc) / ((N - 1) ** 2)


class LinearHSICLoss(nn.Module):
    """
    Lightweight HSIC computation using squared Pearson correlation.
    Complexity: O(N * D_x * D_y) instead of O(N^2)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features, targets):
        """
        features: [N, D_f] (e.g., 1024-dim patch features)
        targets:  [N, D_t] (e.g., 1-dim nadir distances)
        """
        N = features.size(0)

        # Mean-center the features and targets over the batch
        features_c = features - features.mean(dim=0, keepdim=True)
        targets_c = targets - targets.mean(dim=0, keepdim=True)

        features_v = (features_c ** 2).mean(dim=0, keepdim=True)
        targets_v = (targets_c ** 2).mean(dim=0, keepdim=True)

        # Compute the cross-covariance matrix C (Shape: D_f x D_t)
        C = torch.matmul(features_c.t(), targets_c) / (N - 1)

        # Pearson correlation squared: Cov^2 / (Var(F) * Var(T))
        # Minimize the mean squared correlation across all feature/target pairs
        return torch.mean((C ** 2) / (features_v.t() * targets_v + self.eps))


class RFFHSICLoss(nn.Module):
    """
    Lightweight non-linear HSIC using Random Fourier Features (RFF).
    Approximates the RBF kernel without the O(N^2) memory bottleneck.

    Reference: https://arxiv.org/abs/2106.08320
    """
    def __init__(self, feature_dim, target_dim=1, num_rff=128, sigma=1.0):
        super().__init__()
        self.num_rff = num_rff
        
        # Random projection weights (fixed during training)
        # Drawn from a Gaussian distribution to approximate RBF
        self.register_buffer('W_f', torch.randn(feature_dim, num_rff) / sigma)
        self.register_buffer('b_f', torch.rand(num_rff) * 2 * torch.pi)
        
        self.register_buffer('W_t', torch.randn(target_dim, num_rff) / sigma)
        self.register_buffer('b_t', torch.rand(num_rff) * 2 * torch.pi)

    def get_rff(self, x, W, b):
        # Applies the randomized feature mapping: sqrt(2/D) * cos(XW + b)
        projection = torch.matmul(x, W) + b
        return ((2.0 / self.num_rff) ** 0.5) * torch.cos(projection)

    def forward(self, features, targets):
        N = features.size(0)

        # Map features and targets to the RFF space
        Z_f = self.get_rff(features, self.W_f, self.b_f)
        Z_t = self.get_rff(targets, self.W_t, self.b_t)

        # Mean-center the RFF representations
        Z_f_c = Z_f - Z_f.mean(dim=0, keepdim=True)
        Z_t_c = Z_t - Z_t.mean(dim=0, keepdim=True)

        # Compute Linear HSIC on the RFF features
        C = torch.matmul(Z_f_c.t(), Z_t_c) / (N - 1)
        return torch.sum(C ** 2)