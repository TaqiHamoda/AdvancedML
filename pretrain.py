import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import time
import logging
from tqdm import tqdm
import numpy as np

# Import your modules
from src.dataset import MaskingGenerator, TransformedDataset
from src.dino import ConvNeXtV2, ReconstructionHead

logger = logging.getLogger(__name__)


class PreTrainer:
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

        if self.rank == 0:
            logger.info(f"FCMAE Pretraining on {self.device} (World Size: {self.world_size})")

        # --- Data Parameters ---
        self.stride_size = 32
        self.global_crops_size = 288

        # Only 1 global crop for student and teacher, NO local crops
        self.global_crops_number = 1
        self.local_crops_number = 0

        # --- Hyperparameters ---
        self.batch_size = 256 // self.world_size  # Adjust based on your VRAM
        self.epochs = 10
        self.lr = 5e-4
        self.weight_decay = 0.05
        self.mask_ratio = 0.60  # Based on results in paper

        # --- Masking & Dataset ---
        self.mask_generator = MaskingGenerator(
            input_size=self.global_crops_size, 
            stride_size=self.stride_size, 
            mask_ratio=self.mask_ratio
        )

        self.dataset = TransformedDataset(
            global_crops_number=self.global_crops_number,
            local_crops_number=self.local_crops_number,
            global_crops_size=self.global_crops_size,
        )

        if self.is_distributed:
            self.sampler = DistributedSampler(self.dataset, shuffle=True)
        else:
            self.sampler = None

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=TransformedDataset.collate_fn
        )

        self.scaler = torch.amp.GradScaler('cuda')

        # --- Model ---
        class Model(nn.Module):
            def __init__(s):
                super().__init__()

                s.backbone = ConvNeXtV2(in_chans=1)
                s.recon_head = ReconstructionHead(
                    embed_dim=s.backbone.embed_dim,
                    patch_size=self.stride_size,
                    in_chans=1
                )

            def forward(s, x, mask=None, h_grid=None, w_grid=None):
                x_cls, x_patch_flat = s.backbone._fcmae(x, mask=mask)
                return x_cls, s.recon_head(x_patch_flat, h_grid=h_grid, w_grid=w_grid)

        self.model = Model().to(device=self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        # --- Optimizer (Constant LR, No Scheduler) ---
        self.optimizer = optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def train_one_epoch(self, epoch_index):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch_index)

        self.model.train()

        total_loss = 0.0
        for i, batch_imgs in enumerate(self.loader):
            self.optimizer.zero_grad(set_to_none=True)

            # We only need the first global crop
            teacher_crop = batch_imgs['teacher']['global_crops'].to(self.device)
            student_crop = batch_imgs['student']['global_crops'].to(self.device)

            B, C, H, W = student_crop.shape
            mask_grid_h = H // self.stride_size
            mask_grid_w = W // self.stride_size

            # Generate masks for the batch
            masks_list = []
            for _ in range(B):
                masks_list.append(torch.from_numpy(self.mask_generator()).bool())

            masks_flat = torch.stack(masks_list).to(self.device)
            masks_spatial = masks_flat.view(B, mask_grid_h, mask_grid_w)
            active_masks = ~masks_spatial
            upsampled_masks = active_masks.repeat_interleave(self.stride_size, dim=1).repeat_interleave(self.stride_size, dim=2).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                x_cls, reconstructed = self.model(student_crop, mask=active_masks, h_grid=mask_grid_h, w_grid=mask_grid_w)

                mse = (reconstructed - teacher_crop) ** 2
                loss = 0.0 * x_cls.sum() + (mse * upsampled_masks).sum() / upsampled_masks.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if self.rank == 0 and i % 20 == 0:
                logger.info(f"Epoch {epoch_index:03d} [{i:04d}/{len(self.loader)}] | MSE Loss: {loss.item():.6f}")

        return total_loss / len(self.loader)


    def run(self):
        if self.rank == 0:
            logger.info("Starting FCMAE Pretraining...")

        iterator = range(self.epochs)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Pretraining Epochs", total=self.epochs)

        for epoch in iterator:
            avg_loss = self.train_one_epoch(epoch)
            if self.rank == 0:
                logger.info(f"--- Epoch {epoch:03d} Complete | Avg MSE Loss: {avg_loss:.6f} ---")

        # Save weights at the very end
        if self.rank == 0:
            os.makedirs("weights", exist_ok=True)
            save_path = "weights/pretrained.pth"

            torch.save({
                'backbone': self.model.module.backbone.state_dict() if self.is_distributed else self.model.backbone.state_dict(),
                'recon_head': self.model.module.recon_head.state_dict() if self.is_distributed else self.model.recon_head.state_dict(),
            }, save_path)

            logger.info(f"Pretraining complete. Weights saved to {save_path}")

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
            filename=f"logs/pretrain_{time.time()}.log",
            level=logging.INFO
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    trainer = PreTrainer()
    trainer.run()