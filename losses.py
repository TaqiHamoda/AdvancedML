import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, n_iterations=3, center_momentum=0.999):
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
        Cross-entropy between softmax outputs of the teacher and student.
        """
        # 1. Solve for Batch Size (B)
        n_teacher_crops = 2
        batch_size = teacher_output.shape[0] // n_teacher_crops

        # 2. Prepare Student Logits
        # Cast to float() for AMP stability (Good job including this!)
        student_out = (student_output / self.student_temp).float()
        n_student_crops = student_out.shape[0] // batch_size
        student_out = student_out.chunk(n_student_crops)

        # 3. Prepare Teacher Probabilities
        # Using Softmax centering instead of Sinkhorn (Stable for smaller batches)
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(n_teacher_crops)

        total_loss = 0
        n_loss_terms = 0

        # 4. Compute Cross-Entropy Loop
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: 
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        self.update_center(teacher_output)
        return total_loss / n_loss_terms


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