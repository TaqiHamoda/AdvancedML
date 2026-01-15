import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import logging
from tqdm import tqdm
import os, time

# Import your modules
from sonar_data import SonarDataset, SonarDataTransform
from dino import ConvNeXtTiny, DINOHead, MultiCropWrapper
from losses import DINOLoss, GramLoss, KoLeoLoss

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
    output = {'global_crops': [], 'local_crops': []}
    n_global = len(batch[0]['global_crops'])
    n_local = len(batch[0]['local_crops'])

    for i in range(n_global):
        output['global_crops'].append(torch.stack([item['global_crops'][i] for item in batch]))

    for i in range(n_local):
        output['local_crops'].append(torch.stack([item['local_crops'][i] for item in batch]))

    return output

class Trainer:
    def __init__(self):
        # --- 1. Distributed Init ---
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
        
        # --- Hyperparameters ---
        self.batch_size = 64 # Per GPU
        self.base_lr = 0.0005 * self.batch_size * self.world_size / 256  # LINEAR SCALING RULE: Scale LR by world size and batch size
        self.min_lr = 1e-6
        self.weight_decay = 0.04
        self.epochs = 100
        self.warmup_epochs = 10

        self.teacher_temp_start = 0.04
        self.teacher_temp_end = 0.07
        self.teacher_temp_warmup_epochs = 30
        self.momentum_teacher = 0.996
        
        self.w_dino = 1.0
        self.w_gram = 1.0
        self.w_koleo = 0.1
        
        # --- Data & Sampler ---
        dataset = SonarDataset(data_dir="./dataset", ext="*.npy")
        transform = SonarDataTransform(local_crops_number=8)
        
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
            shuffle=(self.sampler is None), # Shuffle handled by sampler if DDP
            sampler=self.sampler,
            num_workers=0, # Decrease workers per GPU
            pin_memory=False,
            drop_last=True,
            collate_fn=dino_collate_fn
        )

        # --- Schedulers ---
        self.teacher_temp_schedule = cosine_scheduler(
            base_value=self.teacher_temp_start,
            final_value=self.teacher_temp_end,
            epochs=self.epochs,
            niter_per_ep=len(self.loader),
            warmup_epochs=self.teacher_temp_warmup_epochs,
            start_warmup_value=self.teacher_temp_start,
        )

        self.lr_schedule = cosine_scheduler(
            base_value=self.base_lr,
            final_value=self.min_lr,
            epochs=self.epochs,
            niter_per_ep=len(self.loader),
            warmup_epochs=self.warmup_epochs,
            start_warmup_value=0,
        )

        self.scaler = torch.amp.GradScaler('cuda')

        # --- Models ---
        student_backbone = ConvNeXtTiny(in_chans=1)
        teacher_backbone = ConvNeXtTiny(in_chans=1)
        embed_dim = student_backbone.embed_dim
        
        student_head = DINOHead(embed_dim, out_dim=65536)
        teacher_head = DINOHead(embed_dim, out_dim=65536)
        
        self.student = MultiCropWrapper(student_backbone, student_head).to(self.device)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(self.device)
        
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict(self.student.state_dict())

        # --- DDP Wrapping ---
        if self.is_distributed:
            # Wrap student. Teacher is NOT wrapped (no gradients).
            self.student = DDP(self.student, device_ids=[self.local_rank])

        # --- Losses ---
        self.dino_loss_fn = DINOLoss(out_dim=65536).to(self.device)
        self.gram_loss_fn = GramLoss().to(self.device)
        self.koleo_loss_fn = KoLeoLoss().to(self.device)

        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            self.get_params_groups(self.student), # Works with DDP wrapped module
            lr=self.base_lr, 
            weight_decay=self.weight_decay 
        )

    def get_params_groups(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim <= 1 or name.endswith(".bias"):
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized, 'weight_decay': self.weight_decay},
                {'params': not_regularized, 'weight_decay': 0.0}]

    def update_teacher_ema(self):
        with torch.no_grad():
            m = self.momentum_teacher
            # Unwrap DDP student for EMA update to avoid name mismatch
            student_model = self.student.module if self.is_distributed else self.student
            
            for param_q, param_k in zip(student_model.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def train_one_epoch(self, epoch_index):
        # CRITICAL: Set epoch for sampler shuffling
        if self.sampler is not None:
            self.sampler.set_epoch(epoch_index)

        for i, batch_imgs in enumerate(self.loader):
            it = len(self.loader) * epoch_index + i
            
            # LR Update
            current_lr = self.lr_schedule[it]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            global_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['global_crops']]
            local_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['local_crops']]
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    teacher_output, teacher_patches_list, _ = self.teacher(global_crops) 
                
                all_crops = global_crops + local_crops
                student_output, student_patches_list, student_cls = self.student(all_crops)
                
                current_teacher_temp = self.teacher_temp_schedule[it]
                loss_dino = self.dino_loss_fn(student_output, teacher_output, current_teacher_temp)
                
                n_global = len(global_crops)
                student_cls_chunked = student_cls.chunk(len(all_crops))
                student_global_cls = torch.cat(student_cls_chunked[:n_global])
                loss_koleo = self.koleo_loss_fn(student_global_cls)
                
                loss_gram = self.gram_loss_fn(student_patches_list[0], teacher_patches_list[0])
                loss = (self.w_dino * loss_dino) + (self.w_gram * loss_gram) + (self.w_koleo * loss_koleo)

            # Optimization outside autocast
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.update_teacher_ema()

            # Log only on Master
            if self.rank == 0 and i % 10 == 0:
                logger.info(f"Epoch {epoch_index} [{i}/{len(self.loader)}] "
                      f"lr: {current_lr:.6f}, Loss: {loss.item():.4f}")

    def run(self):
        if self.rank == 0:
            logger.info("Starting training...")
        
        # Use simple range, tqdm only on master to avoid messed up bars
        iterator = range(self.epochs)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Training Epochs")

        for epoch in iterator:
            self.train_one_epoch(epoch)
            
            # Save only on Master
            if self.rank == 0:
                save_dict = {
                    'epoch': epoch,
                    'student': self.student.module.state_dict() if self.is_distributed else self.student.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(save_dict, f"weights/checkpoint_{epoch}.pth")
        
        if self.is_distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    # Setup logging only on Rank 0 usually, but here we just use basic config
    # A cleaner way is to check env vars before config
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
        logging.basicConfig(level=logging.ERROR) # Silence other processes

    trainer = Trainer()
    trainer.run()