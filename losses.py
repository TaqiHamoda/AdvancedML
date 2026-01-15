import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, n_iterations=3, center_momentum=0.995):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        """
        # Sum the output within the local batch
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # If DDP is enabled, sum across all GPUs
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            # Normalize by total batch size across all GPUs
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            # Single GPU normalization
            batch_center = batch_center / len(teacher_output)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    # Sinkhorn not used since batch size is too small for the unifrom distribution
    # This results in model collapse quickly
    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp):
        Q = torch.exp(teacher_output / teacher_temp).t() 
        K, B = Q.shape[0], Q.shape[1] 
        Q /= torch.sum(Q)

        for _ in range(self.n_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B 
        return Q.t() 

    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Efficient Cross-entropy between teacher and student using torch.einsum.
        """
        n_teacher_crops = 2
        B = teacher_output.shape[0] // n_teacher_crops
        n_student_crops = student_output.shape[0] // B

        # Reshape and cast (float for stability)
        # Shape: (S, B, D)
        s_out = (student_output / self.student_temp).float()
        s_out = s_out.view(n_student_crops, B, -1)
        
        # Shape: (T, B, D)
        t_out = (teacher_output - self.center).float()
        t_out = t_out.view(n_teacher_crops, B, -1)

        # Log-Softmax for Student, Softmax for Teacher
        s_log_probs = F.log_softmax(s_out, dim=-1) 
        t_probs = F.softmax(t_out / teacher_temp, dim=-1)

        # "s b k, t b k -> s t"
        # s: student crops, t: teacher crops, b: batch size, k: embedding dim
        # We sum over batch (b) and embedding (k) to get a (S, T) matrix of total losses
        loss_matrix = -torch.einsum("sbk,tbk->st", s_log_probs, t_probs)

        # We ignore cases where student_crop_idx == teacher_crop_idx (e.g., global1 vs global1)
        # This zeroes out the diagonal elements (0,0), (1,1) etc.
        min_crops = min(n_student_crops, n_teacher_crops)
        loss_matrix = torch.diagonal_scatter(loss_matrix, torch.zeros(min_crops, device=loss_matrix.device))

        # Divide by total number of valid samples (Total pairs * Batch - Diagonal * Batch)
        total_loss = loss_matrix.sum()
        normalization = B * (n_student_crops * n_teacher_crops - min_crops)

        self.update_center(teacher_output)
        return total_loss / normalization


class iBOTPatchLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.995):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_patches, teacher_patches, student_masks_flat, teacher_temp):
        """
        student_patches: (B, N, K)
        teacher_patches: (B, N, K)
        student_masks_flat: (B, N) - Boolean mask where True indicates MASKED
        """
        # Compute Centering and Softmax on the FULL tensors first
        # Student logits (B, N, K)
        s_logits = student_patches / self.student_temp

        # Teacher probabilities (B, N, K) - Centered
        t_out_centered = teacher_patches - self.center
        t_probs = F.softmax(t_out_centered / teacher_temp, dim=-1).detach()

        # Update Center using only the MASKED patches
        mask_bool = student_masks_flat.bool()
        t_masked = teacher_patches[mask_bool]

        if t_masked.numel() > 0:
            self.update_center(t_masked)

        # Calculate Cross Entropy Loss per patch (B, N)
        loss_per_patch = torch.sum(-t_probs * F.log_softmax(s_logits, dim=-1), dim=-1)

        # Apply Mask and Average per Image
        # Zero out losses for unmasked patches
        loss_masked = loss_per_patch * student_masks_flat.float()
        n_masked_per_image = student_masks_flat.sum(dim=1).clamp(min=1.0)
        loss_per_image = loss_masked.sum(dim=1) / n_masked_per_image

        return loss_per_image.mean()


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

        student_gram = torch.bmm(student_patches, student_patches.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))

        student_gram = student_gram.clamp(min=0)
        teacher_gram = teacher_gram.clamp(min=0)

        return self.mse(student_gram, teacher_gram)


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy estimator.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_output):
        x = F.normalize(student_output, dim=-1, p=2)
        
        dists_matrix = torch.cdist(x, x)
        dists_matrix = dists_matrix.clone()
        dists_matrix.fill_diagonal_(float('inf'))

        min_dists, _ = torch.min(dists_matrix, dim=1)
        loss = -torch.log(min_dists + self.eps).mean()

        return loss