import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # If distributed, dist.all_reduce(batch_center)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student.
        """
        # 1. Solve for Batch Size (B)
        # Teacher output is always from the 2 global crops, so Total = 2 * B
        n_teacher_crops = 2
        batch_size = teacher_output.shape[0] // n_teacher_crops
        
        # 2. Prepare Student Logits
        # Split into list of tensors, where each tensor is (B, Dim)
        student_out = student_output / self.student_temp
        n_student_crops = student_out.shape[0] // batch_size
        student_out = student_out.chunk(n_student_crops)

        # 3. Prepare Teacher Probabilities
        # Split into list of tensors, where each tensor is (B, Dim)
        temp = self.teacher_temp
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(n_teacher_crops)

        total_loss = 0
        n_loss_terms = 0
        
        # 4. Compute Cross-Entropy Loop
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # Skip the case where student and teacher view the exact same image
                if v == iq: 
                    continue
                
                # q: (B, Dim), student_out[v]: (B, Dim) -> Loss is valid
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        self.update_center(teacher_output)
        return total_loss / n_loss_terms


class GramLoss(nn.Module):
    """
    Matches the texture/autocorrelation of features between Student and Teacher.
    Crucial for Sonar tile matching.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_patches, teacher_patches):
        # inputs are (Batch, N_patches, Dim)
        
        # Normalize to ensure stability
        student_patches = F.normalize(student_patches, dim=-1)
        teacher_patches = F.normalize(teacher_patches, dim=-1)

        # Compute Gram Matrix: (B, N, D) @ (B, D, N) -> (B, N, N)
        # This represents the relationship between every spatial location and every other location.
        student_gram = torch.bmm(student_patches, student_patches.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))

        # We only want to match the upper triangle to avoid redundancy, but MSE on full matrix is fine/easier
        return self.mse(student_gram, teacher_gram)


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy estimator.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_output):
        # student_output: (Batch, Dim)
        x = F.normalize(student_output, dim=-1, p=2)
        
        # Compute pairwise distances
        dists_matrix = torch.cdist(x, x)
        
        # --- FIX IS HERE ---
        # We must clone the matrix before modifying it in-place.
        # This keeps the original 'dists_matrix' intact for the backward pass of cdist.
        dists_matrix = dists_matrix.clone()
        dists_matrix.fill_diagonal_(float('inf'))
        # -------------------
        
        # Find distance to nearest neighbor for each point
        min_dists, _ = torch.min(dists_matrix, dim=1)
        
        # Maximize entropy
        loss = -torch.log(min_dists + self.eps).mean()
        return loss