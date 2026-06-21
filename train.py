import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

import numpy as np
import logging
from tqdm import tqdm
import os, time
from pathlib import Path

# Import your modules
from src.dataset import MaskingGenerator, TransformedDataset
from src.dino import ConvNeXtV2, DINOHead
from src.losses import DINOLoss, iBOTPatchLoss, GramLoss, KoLeoLoss, HSICLoss, LinearHSICLoss, RFFHSICLoss

logger = logging.getLogger(__name__)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class Trainer:
    def __init__(self):
        self.is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

        if self.is_distributed:
            dist.init_process_group("nccl")
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Only log on master process
        if self.rank == 0:
            logger.info(f"Training on {self.device} (World Size: {self.world_size})")

        # Data Parameters
        self.stride_size = 32
        self.global_crops_size = 224
        self.local_crops_size = 96
        self.global_crops_number = 2
        self.local_crops_number = 8

        # --- Hyperparameters ---
        self.output_dim = 4096  # Number of prototypes outputted by DINO
        self.batch_size = 30  # Max possible per GPU
        self.effective_batch_size = 8192 // self.world_size  # Desired batch size per GPU
        self.accum_iter = self.effective_batch_size // self.batch_size  # Number of gradient accumulation steps
        self.base_lr = 5e-5 * (self.world_size * self.effective_batch_size / 1024) ** 0.5  # Square root scaling
        self.weight_decay = 0.04
        self.epochs = 100
        self.warmup_epochs = self.epochs // 10

        self.center_momentum = 1.0 - (1.0 - 0.996) / self.accum_iter  # Scale with the gradient accumulation steps

        self.teacher_temp_start = 0.04
        self.teacher_temp_end = 0.07
        self.momentum_teacher_start = 0.996
        self.momentum_teacher_end = 1.0

        self.w_dino = 1.0
        self.w_ibot = 1.0
        self.w_gram = 0.5
        self.w_hsic = 0.0
        self.w_koleo = 0.01

        # --- Masking, Data & Sampler ---
        self.mask_generator = MaskingGenerator(input_size=self.global_crops_size, stride_size=self.stride_size, mask_ratio=0.5)
        self.dataset = TransformedDataset(
            global_crops_number=self.global_crops_number,
            local_crops_number=self.local_crops_number,
            global_crops_size=self.global_crops_size,
            local_crops_size=self.local_crops_size
        )

        # DISTRIBUTED SAMPLER
        if self.is_distributed:
            self.sampler = DistributedSampler(self.dataset, shuffle=True)
        else:
            self.sampler = None

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),  # Shuffle handled by sampler if DDP
            sampler=self.sampler,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
            collate_fn=TransformedDataset.collate_fn
        )

        # --- Schedulers ---
        self.effective_niter_per_ep = (len(self.loader) + self.accum_iter - 1) // self.accum_iter

        self.teacher_temp_schedule = cosine_scheduler(
            base_value=self.teacher_temp_start,
            final_value=self.teacher_temp_end,
            epochs=self.epochs,
            niter_per_ep=self.effective_niter_per_ep,
            warmup_epochs=self.warmup_epochs,
            start_warmup_value=self.teacher_temp_start,
        )

        self.lr_schedule = cosine_scheduler(
            base_value=self.base_lr,
            final_value=self.base_lr,
            epochs=self.epochs,
            niter_per_ep=self.effective_niter_per_ep,
            warmup_epochs=self.warmup_epochs,
            start_warmup_value=0,
        )

        self.momentum_schedule = cosine_scheduler(
            base_value=self.momentum_teacher_start,
            final_value=self.momentum_teacher_end,
            epochs=self.epochs,
            niter_per_ep=self.effective_niter_per_ep,
        )

        self.scaler = torch.amp.GradScaler('cuda')

        # --- Models ---
        self.student = ConvNeXtV2(in_chans=1).to(self.device)
        self.teacher = ConvNeXtV2(in_chans=1).to(self.device)
        embed_dim = self.student.embed_dim

        self.student_dino_head = DINOHead(embed_dim, out_dim=self.output_dim).to(self.device)
        self.teacher_dino_head = DINOHead(embed_dim, out_dim=self.output_dim).to(self.device)

        self.teacher.eval()  # Teacher is not trained with gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict(self.student.state_dict())

        self.student_ibot_head = DINOHead(embed_dim, out_dim=self.output_dim).to(self.device)
        self.teacher_ibot_head = DINOHead(embed_dim, out_dim=self.output_dim).to(self.device)

        self.teacher_ibot_head.eval()
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
        for p in self.teacher_ibot_head.parameters(): p.requires_grad = False

        # --- DDP Wrapping ---
        if self.is_distributed:
            # Wrap student. Teacher is NOT wrapped (no gradients).
            self.student = DDP(self.student, device_ids=[self.local_rank])
            self.student_dino_head = DDP(self.student_dino_head, device_ids=[self.local_rank])
            self.student_ibot_head = DDP(self.student_ibot_head, device_ids=[self.local_rank])

        # --- Losses ---
        self.dino_loss_fn = DINOLoss(out_dim=self.output_dim, center_momentum=self.center_momentum).to(self.device)
        self.ibot_loss_fn = iBOTPatchLoss(out_dim=self.output_dim, center_momentum=self.center_momentum).to(self.device)
        self.gram_loss_fn = GramLoss().to(self.device)
        self.hsic_loss_fn = RFFHSICLoss(feature_dim=embed_dim).to(self.device)
        self.koleo_loss_fn = KoLeoLoss().to(self.device)

        # --- Optimizer ---
        params_to_optimize = self.get_params_groups(self.student)
        params_to_optimize += self.get_params_groups(self.student_dino_head)
        params_to_optimize += self.get_params_groups(self.student_ibot_head)
        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=self.base_lr, 
            weight_decay=self.weight_decay 
        )

    def get_params_groups(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim <= 1 or name.endswith(".bias") or "last_layer" in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized, 'weight_decay': self.weight_decay},
                {'params': not_regularized, 'weight_decay': 0.0}]

    def train_one_epoch(self, epoch_index):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch_index)

        self.optimizer.zero_grad(set_to_none=True) # Ensure gradients are zero at start
        for i, batch_imgs in enumerate(self.loader):
            optim_step = i // self.accum_iter
            it = self.effective_niter_per_ep * epoch_index + optim_step

            # LR Update
            current_lr = self.lr_schedule[it]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                # Only apply weight decay to the regularized group (index 0)
                if param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = self.weight_decay

            # Shape of each item is (B * N_patches, C, W, H)
            teacher_global_crops = batch_imgs['teacher']['global_crops'].to(self.device)
            student_global_crops = batch_imgs['student']['global_crops'].to(self.device)
            student_local_crops = batch_imgs['student']['local_crops'].to(self.device)
            distance_global_crops = batch_imgs['distances']['global_crops'].to(self.device)

            masks_list = []
            for _ in range(student_global_crops.shape[0]):
                masks_list.append(torch.from_numpy(self.mask_generator()).bool())  # True means "Drop" here

            masks_spatial = torch.stack(masks_list).to(self.device)
            active_masks = ~masks_spatial  # Original iBOT mask (B * N_patches, C, W, H). True = Keep.

            is_accumulating = ((i + 1) % self.accum_iter != 0) and ((i + 1) != len(self.loader))
            if self.is_distributed and is_accumulating:
                ctx_student = self.student.no_sync()
                ctx_dino = self.student_dino_head.no_sync()
                ctx_ibot = self.student_ibot_head.no_sync()
            else:
                ctx_student = nullcontext()
                ctx_dino = nullcontext()
                ctx_ibot = nullcontext()

            with ctx_student, ctx_dino, ctx_ibot:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        # Teacher gets no masks
                        teacher_cls, teacher_patches = self.teacher(teacher_global_crops, mask=None)

                        scale_factor = int(teacher_patches.shape[-2] ** 0.5) // masks_spatial.shape[-1]
                        upsampled_masks = masks_spatial.repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
                        upsampled_masks = upsampled_masks.flatten(1)  # (B, H, W) -> (B, H*W)

                        t_patches = teacher_patches[upsampled_masks]

                        t_dino_out = self.teacher_dino_head(teacher_cls)  # (B * N_patches, output_dim)
                        t_ibot_out = self.teacher_ibot_head(t_patches)  # (Total_Masked_Tokens, K)

                    student_global_cls, student_global_patches = self.student(student_global_crops, mask=active_masks)

                    s_global_patches = student_global_patches[upsampled_masks]
                    s_dino_global_out = self.student_dino_head(student_global_cls)  # (B * N_patches, output_dim)
                    s_ibot_out = self.student_ibot_head(s_global_patches)  # (Total_Masked_Tokens, K)

                    student_local_cls, student_local_patches = self.student(student_local_crops, mask=None)
                    s_dino_local_out = self.student_dino_head(student_local_cls)  # (B * N_patches, output_dim)

                    # Interleave the student outputs per image before calculating loss
                    s_dino_out = torch.cat([
                        s_dino_global_out.view(self.batch_size, self.global_crops_number, -1),
                        s_dino_local_out.view(self.batch_size, self.local_crops_number, -1)
                    ], dim=1)
                    s_dino_out = s_dino_out.reshape(-1, s_dino_out.shape[-1])

                    current_teacher_temp = self.teacher_temp_schedule[it]

                    loss_dino = self.dino_loss_fn(s_dino_out, t_dino_out, current_teacher_temp, n_teacher_crops=self.global_crops_number)
                    loss_ibot = self.ibot_loss_fn(s_ibot_out, t_ibot_out, current_teacher_temp)
                    loss_koleo = self.koleo_loss_fn(student_global_cls[::2])  # Pass ONLY the even rows (unique independent images)

                    # loss_gram = self.gram_loss_fn(student_patches_list[0], teacher_patches_list[0])
                    # loss = (self.w_dino * loss_dino) + (self.w_ibot * loss_ibot) + (self.w_gram * loss_gram) + (self.w_koleo * loss_koleo)
                    loss = (self.w_dino * loss_dino) + (self.w_ibot * loss_ibot) + (self.w_koleo * loss_koleo)
                    loss = loss / self.accum_iter  # Normalize loss to account for accumulation

                # Log only on Master
                if self.rank == 0 and i % self.accum_iter == 0:
                    logger.info(f"Epoch {epoch_index:03d} [{i:04d}/{len(self.loader)}] "
                        f"lr: {current_lr:.6f}, t: {self.teacher_temp_schedule[it]:.4f}, m: {self.momentum_schedule[it]:.4f}, "
                        f"DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, KoLeo: {loss_koleo.item():.4f}")
                        # f"DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, Gram: {loss_gram.item():.4f}, KoLeo: {loss_koleo.item():.4f}")

                # Backward pass (Accumulates gradients into .grad attributes)
                self.scaler.scale(loss).backward()

            # Manually delete heavy tensors to free VRAM for the next iteration
            # del loss, loss_ibot, loss_gram, loss_koleo
            del loss, loss_dino, loss_ibot, loss_koleo
            del s_dino_out, t_dino_out, s_ibot_out, t_ibot_out
            del student_local_cls, student_local_patches, s_dino_local_out
            del s_dino_global_out, s_global_patches, student_global_cls, student_global_patches
            del t_patches, upsampled_masks, teacher_patches, teacher_cls
            del distance_global_crops, student_local_crops, student_global_crops, teacher_global_crops
            del active_masks, masks_list

            if ((i + 1) % self.accum_iter == 0) or ((i + 1) == len(self.loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.student_dino_head.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.student_ibot_head.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                with torch.no_grad():
                    m = self.momentum_schedule[it]

                    student_model = self.student.module if self.is_distributed else self.student
                    for param_q, param_k in zip(student_model.parameters(), self.teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                    student_dino_head = self.student_dino_head.module if self.is_distributed else self.student_dino_head
                    for p_s, p_t in zip(student_dino_head.parameters(), self.teacher_dino_head.parameters()):
                        p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)

                    student_ibot_head = self.student_ibot_head.module if self.is_distributed else self.student_ibot_head
                    for p_s, p_t in zip(student_ibot_head.parameters(), self.teacher_ibot_head.parameters()):
                        p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)


    def load_checkpoint(self, resume_path):
        checkpoint_path = resume_path / "checkpoint_latest.pth"
        pretrained_path = resume_path / "pretrained.pth"

        epoch = -1
        if checkpoint_path.exists():
            if self.rank == 0: logger.info(f"Loading checkpoint from {checkpoint_path}")

            # Load on CPU first to avoid OOM, then move to device
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if self.is_distributed:
                self.student.module.load_state_dict(checkpoint['student'])
                self.student_dino_head.module.load_state_dict(checkpoint['student_dino_head'])
                self.student_ibot_head.module.load_state_dict(checkpoint['student_ibot_head'])
            else:
                self.student.load_state_dict(checkpoint['student'])
                self.student_dino_head.load_state_dict(checkpoint['student_dino_head'])
                self.student_ibot_head.load_state_dict(checkpoint['student_ibot_head'])

            self.teacher.load_state_dict(checkpoint['teacher'])
            self.teacher_dino_head.load_state_dict(checkpoint['teacher_dino_head'])
            self.teacher_ibot_head.load_state_dict(checkpoint['teacher_ibot_head'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])

            self.dino_loss_fn.load_state_dict(checkpoint['dino_loss'])
            self.ibot_loss_fn.load_state_dict(checkpoint['ibot_loss'])

            epoch = checkpoint['epoch'] + 1

            # Free memory
            del checkpoint
            torch.cuda.empty_cache()
        elif pretrained_path.exists():
            if self.rank == 0: logger.info(f"Loading pretrained weights")

            # Load on CPU first to avoid OOM, then move to device
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if self.is_distributed:
                self.student.module.load_state_dict(checkpoint['backbone'])
            else:
                self.student.load_state_dict(checkpoint['backbone'])

            self.teacher.load_state_dict(checkpoint['backbone'])

            epoch = 0

            # Free memory
            del checkpoint
            torch.cuda.empty_cache()
        else:
            if self.rank == 0: logger.warning(f"Checkpoint not found at {checkpoint_path}")

        return epoch

    def run(self, resume_path=None):
        start_epoch = 0

        loaded_data = self.load_checkpoint(resume_path)
        if loaded_data > -1:
            start_epoch = loaded_data
            if self.rank == 0: 
                logger.info(f"Resuming training from epoch {start_epoch}")

        if self.rank == 0:
            logger.info(f"Model collapse happens at DINO/iBOT loss value: ln({self.output_dim}) ~ {np.log(self.output_dim):.2f}")
            logger.info("Starting training...")

        # Use simple range, tqdm only on master to avoid messed up bars
        iterator = range(start_epoch, self.epochs)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Training Epochs", initial=start_epoch, total=self.epochs)

        self.student.train()
        self.student_dino_head.train()
        self.student_ibot_head.train()

        for epoch in iterator:
            self.train_one_epoch(epoch)
            if self.rank == 0:
                save_dict = {
                    'epoch': epoch,
                    'student': self.student.module.state_dict() if self.is_distributed else self.student.state_dict(),
                    'student_dino_head': self.student_dino_head.module.state_dict() if self.is_distributed else self.student_dino_head.state_dict(),
                    'student_ibot_head': self.student_ibot_head.module.state_dict() if self.is_distributed else self.student_ibot_head.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'teacher_dino_head': self.teacher_dino_head.state_dict(),
                    'teacher_ibot_head': self.teacher_ibot_head.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.scaler.state_dict(),
                    'dino_loss': self.dino_loss_fn.state_dict(),
                    'ibot_loss': self.ibot_loss_fn.state_dict()
                }
                torch.save(save_dict, f"weights/checkpoint_{epoch}.pth")
                torch.save(save_dict, f"weights/checkpoint_latest.pth")

        if self.is_distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("weights", exist_ok=True)
        logging.basicConfig(
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            filename=f"logs/{time.time()}.log",
            level=logging.INFO
        )
    else:
        logging.basicConfig(level=logging.ERROR)  # Silence other processes

    trainer = Trainer()

    resume_path = Path("weights/")
    if resume_path.exists():
        trainer.run(resume_path=resume_path)
    else:
        trainer.run()
