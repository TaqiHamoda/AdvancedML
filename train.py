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

# Import your modules
from src.dataset import MaskingGenerator, SonarDataset, SonarDataTransform
from src.dino import ConvNeXtV2, DINOHead, MultiCropWrapper
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


def dino_collate_fn(batch):
    output = {
        'teacher': {'global_crops': [], 'local_crops': []},
        'student': {'global_crops': [], 'local_crops': []},
        'distances': {'global_crops': [], 'local_crops': []},
    }

    # Collate standard crops
    for model in output.keys():
        for crop in output[model].keys():
            for i in range(len(batch[0][model][crop])):
                output[model][crop].append(torch.stack([item[model][crop][i] for item in batch]))

    return output


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
        self.tile_size = 384
        self.global_crop_size = 224
        self.local_crop_size = 96
        self.local_crops_number = 8

        # --- Hyperparameters ---
        self.output_dim = 4096  # Number of prototypes outputted by DINO
        self.batch_size = 30  # Max possible per GPU
        self.effective_batch_size = 8192 // self.world_size  # Desired batch size per GPU
        self.accum_iter = self.effective_batch_size // self.batch_size  # Number of gradient accumulation steps
        self.base_lr = 5e-4 * (self.world_size * self.effective_batch_size / 1024) ** 0.5  # Square root scaling
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
        self.w_koleo = 0.1

        # --- Masking, Data & Sampler ---
        self.mask_generator = MaskingGenerator(input_size=self.global_crop_size, stride_size=self.stride_size, mask_ratio=0.5)
        dataset = SonarDataset(tile_size=self.tile_size)
        transform = SonarDataTransform(local_crops_number=self.local_crops_number, global_crops_size=self.global_crop_size, local_crops_size=self.local_crop_size)
        
        # Wrapper class to apply transform on the fly
        class TransformedDataset(torch.utils.data.Dataset):
            def __init__(self, ds, tf): self.ds = ds; self.tf = tf
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx): return self.tf(self.ds[idx])

        self.dataset = TransformedDataset(dataset, transform)

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
            collate_fn=dino_collate_fn
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
        student_backbone = ConvNeXtV2(in_chans=1)
        teacher_backbone = ConvNeXtV2(in_chans=1)
        embed_dim = student_backbone.embed_dim

        student_head = DINOHead(embed_dim, out_dim=self.output_dim)
        teacher_head = DINOHead(embed_dim, out_dim=self.output_dim)

        self.student = MultiCropWrapper(student_backbone, student_head).to(self.device)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(self.device)

        self.teacher.eval()  # Teacher is not trained with gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict(self.student.state_dict())

        # Add iBOT Heads (Separate from DINO Head)
        # They project patch tokens (embed_dim) -> prototypes (output_dim)
        student_ibot_head = DINOHead(embed_dim, out_dim=self.output_dim)
        teacher_ibot_head = DINOHead(embed_dim, out_dim=self.output_dim)

        self.student_ibot_head = student_ibot_head.to(self.device)
        self.teacher_ibot_head = teacher_ibot_head.to(self.device)
        self.teacher_ibot_head.eval()
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        for p in self.teacher_ibot_head.parameters(): p.requires_grad = False

        # --- DDP Wrapping ---
        if self.is_distributed:
            # Wrap student. Teacher is NOT wrapped (no gradients).
            self.student = DDP(self.student, device_ids=[self.local_rank])
            self.student_ibot_head = DDP(self.student_ibot_head, device_ids=[self.local_rank])

        # --- Losses ---
        self.dino_loss_fn = DINOLoss(out_dim=self.output_dim, center_momentum=self.center_momentum).to(self.device)
        self.ibot_loss_fn = iBOTPatchLoss(out_dim=self.output_dim, center_momentum=self.center_momentum).to(self.device)
        self.gram_loss_fn = GramLoss().to(self.device)
        self.hsic_loss_fn = RFFHSICLoss(feature_dim=embed_dim).to(self.device)
        self.koleo_loss_fn = KoLeoLoss().to(self.device)

        # --- Optimizer ---
        params_to_optimize = self.get_params_groups(self.student)
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

            teacher_global_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['teacher']['global_crops']]
            student_global_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['student']['global_crops']]
            student_local_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['student']['local_crops']]
            distance_global_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['distances']['global_crops']]

            B = student_global_crops[0].shape[0]
            masks_list = []
            for _ in range(B * 2): 
                m = self.mask_generator() # 1/True usually means "Drop" here
                masks_list.append(torch.from_numpy(m).bool())

            # Original iBOT mask (2*B, N_patches). True = Dropped.
            masks = torch.stack(masks_list).to(self.device)

            mask_grid_h = student_global_crops[0].shape[-2] // self.stride_size
            mask_grid_w = student_global_crops[0].shape[-1] // self.stride_size

            masks_spatial = masks.view(-1, mask_grid_h, mask_grid_w)
            active_masks = ~masks_spatial 

            # Split the active mask into two chunks for the two global crops
            active_masks_chunked = list(torch.chunk(active_masks, 2, dim=0))

            all_student_crops = student_global_crops + student_local_crops
            all_student_masks = active_masks_chunked + [None] * len(student_local_crops)  # Local crops don't get masked

            masks_flat = masks.view(masks.shape[0], -1).bool()

            is_accumulating = ((i + 1) % self.accum_iter != 0) and ((i + 1) != len(self.loader))
            if self.is_distributed and is_accumulating:
                ctx_student = self.student.no_sync()
                ctx_ibot = self.student_ibot_head.no_sync()
            else:
                ctx_student = nullcontext()
                ctx_ibot = nullcontext()

            with ctx_student, ctx_ibot:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        # Teacher gets no masks
                        teacher_output, teacher_patches_list, _ = self.teacher(teacher_global_crops, masks=None)
                        
                        t_patches = torch.cat(teacher_patches_list, dim=0)  # (2*B, N, D)
                        t_patches_masked = t_patches[masks_flat]  # (Total_Masked_Tokens, D)
                        t_ibot_out = self.teacher_ibot_head(t_patches_masked)  # (Total_Masked_Tokens, K)

                    # Student gets the crops AND the masks
                    student_output, student_patches_list, student_cls = self.student(all_student_crops, masks=all_student_masks)

                    current_teacher_temp = self.teacher_temp_schedule[it]
                    loss_dino = self.dino_loss_fn(student_output, teacher_output, current_teacher_temp)

                    # iBOT Loss (Patch tokens)
                    # Select only the global crop patches from student output (first 2 items)
                    s_global_patches = student_patches_list[0]  # (2*B, N, D)
                    s_patches_masked = s_global_patches[masks_flat]  # (Total_Masked_Tokens, D)
                    s_ibot_out = self.student_ibot_head(s_patches_masked)  # (Total_Masked_Tokens, K)
                    loss_ibot = self.ibot_loss_fn(
                        s_ibot_out,
                        t_ibot_out,
                        masks,
                        current_teacher_temp
                    )

                    B_features, N_patches, D = s_global_patches.shape
                    num_crops = B_features // student_global_crops[0].shape[0]
                    dist_crops_cat = torch.cat(distance_global_crops[:num_crops], dim=0) # (B_features, 1, 288, 288)

                    grid_size = int(round(N_patches ** 0.5))
                    if grid_size * grid_size == N_patches:
                        patch_dist = F.adaptive_avg_pool2d(dist_crops_cat, (grid_size, grid_size))
                    else:
                        patch_dist = F.adaptive_avg_pool2d(dist_crops_cat, (N_patches, 1))

                    patch_dist_flat = patch_dist.view(B_features, -1, 1) # (B_features, N_patches, 1)

                    loss_hsic = self.hsic_loss_fn(
                        s_global_patches.view(-1, D), 
                        patch_dist_flat.view(-1, 1).to(s_global_patches.dtype)
                    )

                    student_cls_chunked = student_cls.chunk(len(all_student_crops))
                    loss_koleo = self.koleo_loss_fn(student_cls_chunked[0])  # Pass ONLY the first global crop (unique independent images)

                    # loss_gram = self.gram_loss_fn(student_patches_list[0], teacher_patches_list[0])
                    # loss = (self.w_dino * loss_dino) + (self.w_ibot * loss_ibot) + (self.w_gram * loss_gram) + (self.w_koleo * loss_koleo)
                    loss = (self.w_dino * loss_dino) + (self.w_ibot * loss_ibot) + (self.w_hsic * loss_hsic) + (self.w_koleo * loss_koleo)
                    loss = loss / self.accum_iter  # Normalize loss to account for accumulation

                # Log only on Master
                if self.rank == 0 and i % self.accum_iter == 0:
                    logger.info(f"Epoch {epoch_index:03d} [{i:04d}/{len(self.loader)}] "
                        f"lr: {current_lr:.6f}, t: {self.teacher_temp_schedule[it]:.4f}, m: {self.momentum_schedule[it]:.4f}, "
                        f"DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, HSIC: {loss_hsic.item():.4f}, KoLeo: {loss_koleo.item():.4f}")
                        # f"DINO: {loss_dino.item():.4f}, iBOT: {loss_ibot.item():.4f}, Gram: {loss_gram.item():.4f}, KoLeo: {loss_koleo.item():.4f}")

                # Backward pass (Accumulates gradients into .grad attributes)
                self.scaler.scale(loss).backward()

            # Manually delete heavy tensors to free VRAM for the next iteration
            # del loss, loss_ibot, loss_gram, loss_koleo
            del loss, loss_ibot, loss_koleo
            del student_output, teacher_output
            del s_ibot_out, t_ibot_out
            del teacher_global_crops, student_global_crops, student_local_crops, masks
            del student_patches_list, teacher_patches_list
            del student_cls, masks_flat
            del all_student_crops, all_student_masks

            if ((i + 1) % self.accum_iter == 0) or ((i + 1) == len(self.loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
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

                    student_ibot_head = self.student_ibot_head.module if self.is_distributed else self.student_ibot_head
                    for p_s, p_t in zip(student_ibot_head.parameters(), self.teacher_ibot_head.parameters()):
                        p_t.data.mul_(m).add_((1 - m) * p_s.detach().data)


    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            if self.rank == 0: logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return -1

        if self.rank == 0: logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load on CPU first to avoid OOM, then move to device
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if self.is_distributed:
            self.student.module.load_state_dict(checkpoint['student'])
            self.student_ibot_head.module.load_state_dict(checkpoint['student_ibot_head'])
        else:
            self.student.load_state_dict(checkpoint['student'])
            self.student_ibot_head.load_state_dict(checkpoint['student_ibot_head'])

        self.teacher.load_state_dict(checkpoint['teacher'])
        self.teacher_ibot_head.load_state_dict(checkpoint['teacher_ibot_head'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])

        epoch = checkpoint['epoch']

        # Free memory
        del checkpoint
        torch.cuda.empty_cache()

        return epoch

    def run(self, resume_path=None):
        start_epoch = 0

        loaded_data = self.load_checkpoint(resume_path)
        if loaded_data > -1:
            start_epoch = loaded_data + 1
            if self.rank == 0: 
                logger.info(f"Resuming training from epoch {start_epoch}")

        if self.rank == 0:
            logger.info(f"Model collapse happens at DINO/iBOT loss value: ln({self.output_dim}) ~ {np.log(self.output_dim):.2f}")
            logger.info("Starting training...")

        # Use simple range, tqdm only on master to avoid messed up bars
        iterator = range(start_epoch, self.epochs)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Training Epochs", initial=start_epoch, total=self.epochs)

        for epoch in iterator:
            self.train_one_epoch(epoch)
            if self.rank == 0:
                save_dict = {
                    'epoch': epoch,
                    'student': self.student.module.state_dict() if self.is_distributed else self.student.state_dict(),
                    'student_ibot_head': self.student_ibot_head.module.state_dict() if self.is_distributed else self.student_ibot_head.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'teacher_ibot_head': self.teacher_ibot_head.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.scaler.state_dict(),
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

    resume_file = "weights/checkpoint_latest.pth"
    if os.path.exists(resume_file):
        trainer.run(resume_path=resume_file)
    else:
        trainer.run()
