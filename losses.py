import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SinkhornKnopp(nn.Module):
    def __init__(self, student_temp=0.1, n_iterations=3, eps=1e-6):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations
        self.eps = eps

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        Q = torch.exp(teacher_output / teacher_temp).t() 
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
        return Q.t()


class DINOLoss(nn.Module):
    """
    DINOv1-style Loss: Centering + Softmax. 
    More stable for small batch sizes (B < K).
    """
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Args:
            student_output: (S * B, D) student logits
            teacher_output: (T * B, D) teacher logits
            teacher_temp: scalar temperature
        """
        n_teacher_crops = 2
        B = teacher_output.shape[0] // n_teacher_crops

        # Center the Teacher Logits
        # teacher_output is (T*B, D)
        teacher_out_centered = teacher_output - self.center

        # Apply Softmax to Teacher
        t_probs = F.softmax(teacher_out_centered / teacher_temp, dim=-1).detach()

        # Student Log-Softmax
        s_out = student_output / self.student_temp
        s_log_probs = F.log_softmax(s_out, dim=-1)

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

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center with Exponential Moving Average (EMA).
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class iBOTPatchLoss(SinkhornKnopp):
    def forward(self, student_patches, teacher_patches, masks, teacher_temp):
        # Student Log-Softmax
        s_logits = student_patches / self.student_temp
        s_log_probs = F.log_softmax(s_logits, dim=-1)

        # Teacher Sinkhorn (replacing simple softmax)
        t_probs = self.sinkhorn_knopp_teacher(teacher_patches, teacher_temp).detach()

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

        student_gram = torch.bmm(student_patches, student_patches.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))

        student_gram = student_gram.clamp(min=0)
        teacher_gram = teacher_gram.clamp(min=0)

        return self.mse(student_gram, teacher_gram)


class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.pdist = nn.PairwiseDistance(2, eps=eps)

    def forward(self, student_output):
        x = F.normalize(student_output, dim=-1, p=2)

        # Gather all features from other GPUs
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

        return -torch.log(distances + self.eps).mean()