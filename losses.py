import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, n_iterations=3, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # If distributed, dist.all_reduce(batch_center)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    # Sinkhorn not used since batch size is too small for the unifrom distribution
    # This results in model collapse quickly
    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp):
        """
        Applies Sinkhorn-Knopp normalization to the teacher output.
        This enforces a uniform distribution of the teacher's output across the batch.
        """        
        # teacher_output: [Batch * n_crops, out_dim]
        # Q is K-by-B for consistency with the algorithm notations
        Q = torch.exp(teacher_output / teacher_temp).t() # (out_dim, Batch_Total)

        K, B = Q.shape[0], Q.shape[1] # Total batch size (local)
        Q /= torch.sum(Q)

        for _ in range(self.n_iterations):
            # Normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K

            # Normalize each column: total weight per sample must be 1/B_total
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # The columns must sum to 1 so that Q is an assignment probability
        return Q.t() # Return to (Batch, out_dim)

    # def forward(self, student_output, teacher_output, teacher_temp):
    #     """
    #     Cross-entropy between softmax outputs of the teacher and student.
    #     """
    #     # Teacher output is always from the 2 global crops, so Total = 2 * B
    #     n_teacher_crops = 2
        
    #     # Split teacher outputs back into per-crop views for the loop
    #     teacher_out = self.sinkhorn_knopp_teacher(teacher_output, teacher_temp)
    #     teacher_out = teacher_out.detach().chunk(n_teacher_crops)

    #     # Student_output contains Global + Local crops
    #     # Split into list of tensors, where each tensor is (B, Dim)
    #     batch_size = teacher_output.shape[0] // n_teacher_crops
    #     student_out = student_output / self.student_temp
    #     n_student_crops = student_out.shape[0] // batch_size
    #     student_out = student_out.chunk(n_student_crops)

    #     # Compute cross-entropy loss between teacher and student
    #     total_loss, n_loss_terms = 0, 0
    #     for iq, q in enumerate(teacher_out):
    #         for v in range(len(student_out)):
    #             # Skip the case where student and teacher view the exact same image
    #             if v == iq: 
    #                 continue

    #             # q is the target probability from teacher (after Sinkhorn)
    #             # student_out[v] are the raw logits from student
    #             loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
    #             total_loss += loss.mean()

    #             n_loss_terms += 1

    #     return total_loss / n_loss_terms

    def forward(self, student_output, teacher_output, teacher_temp):
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
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
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
        # Normalize to ensure stability
        student_patches = F.normalize(student_patches, dim=-1)
        teacher_patches = F.normalize(teacher_patches, dim=-1)

        # Compute Gram Matrix: (B, N, D) @ (B, D, N) -> (B, N, N)
        # This represents the relationship between every spatial location and every other location.
        student_gram = torch.bmm(student_patches, student_patches.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))

        # We clamp negative values to 0. We don't care if features are opposites. Only match positive patterns
        student_gram = student_gram.clamp(min=0)
        teacher_gram = teacher_gram.clamp(min=0)

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
        dists_matrix = dists_matrix.clone()
        dists_matrix.fill_diagonal_(float('inf'))

        # Find distance to nearest neighbor for each point and maximize the entropy
        min_dists, _ = torch.min(dists_matrix, dim=1)
        loss = -torch.log(min_dists + self.eps).mean()

        return loss