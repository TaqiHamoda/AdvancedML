import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SinkhornKnopp(nn.Module):
    def __init__(self, student_temp=0.1, n_iterations=3):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations

    # Reuse the same Sinkhorn logic (or inherit from a shared base class)
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
        Q /= sum_Q

        for _ in range(self.n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B 
        return Q.t()


class DINOLoss(SinkhornKnopp):
    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Args:
            student_output: (S * B, D) student logits
            teacher_output: (T * B, D) teacher logits
            teacher_temp: scalar temperature
        """
        n_teacher_crops = 2
        B = teacher_output.shape[0] // n_teacher_crops
        n_student_crops = student_output.shape[0] // B

        # Student: Log-Softmax with temperature
        s_out = (student_output / self.student_temp).float()
        s_out = s_out.view(n_student_crops, B, -1)
        s_log_probs = F.log_softmax(s_out, dim=-1) 

        # Teacher: Sinkhorn-Knopp centering + sharpening
        t_probs = self.sinkhorn_knopp_teacher(teacher_output, teacher_temp)
        t_probs = t_probs.view(n_teacher_crops, B, -1).detach()

        # Cross-Entropy Loss
        # "s b k, t b k -> s t" sum over batch and embedding dim
        loss_matrix = -torch.einsum("sbk,tbk->st", s_log_probs, t_probs)

        # Remove diagonal (same crop comparison)
        min_crops = min(n_student_crops, n_teacher_crops)
        loss_matrix = torch.diagonal_scatter(loss_matrix, torch.zeros(min_crops, device=loss_matrix.device))

        total_loss = loss_matrix.sum()
        normalization = B * (n_student_crops * n_teacher_crops - min_crops)

        return total_loss / normalization


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