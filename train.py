import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time

# Import your modules
from sonar_data import SonarDataset, SonarDataTransform
from dino import ConvNeXtTiny, DINOHead, MultiCropWrapper
from losses import DINOLoss, GramLoss, KoLeoLoss

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on {self.device}")
        
        # --- Hyperparameters ---
        self.batch_size = 32 # Adjust based on your GPU memory (32GB can likely handle 64+)
        self.base_lr = 0.0005
        self.weight_decay = 0.04
        self.epochs = 100
        self.momentum_teacher = 0.996 # Standard DINO EMA
        
        # Loss Weights
        self.w_dino = 1.0
        self.w_gram = 0.5  # High weight for texture matching
        self.w_koleo = 0.1
        
        # --- Data ---
        self.dataset = SonarDataset(data_dir="./dataset", ext="*.npy")
        self.transform = SonarDataTransform(local_crops_number=8)
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            drop_last=True
        )

        # --- Models ---
        student_backbone = ConvNeXtTiny(in_chans=1)
        teacher_backbone = ConvNeXtTiny(in_chans=1)
        
        embed_dim = student_backbone.embed_dim
        
        student_head = DINOHead(embed_dim, out_dim=65536) # 65k prototypes is standard
        teacher_head = DINOHead(embed_dim, out_dim=65536)
        
        self.student = MultiCropWrapper(student_backbone, student_head).to(self.device)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(self.device)
        
        # Teacher does not require gradients (updated via EMA)
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())

        # --- Losses ---
        self.dino_loss_fn = DINOLoss(out_dim=65536).to(self.device)
        self.gram_loss_fn = GramLoss().to(self.device)
        self.koleo_loss_fn = KoLeoLoss().to(self.device)

        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            self.student.parameters(), 
            lr=self.base_lr, 
            weight_decay=self.weight_decay
        )

    def update_teacher_ema(self):
        # Apply EMA: teacher = m * teacher + (1 - m) * student
        with torch.no_grad():
            m = self.momentum_teacher
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def train_one_epoch(self, epoch_index):
        total_loss = 0
        
        for i, batch_imgs in enumerate(self.loader):
            # Apply transforms on the fly
            # Note: Transformations are usually done in Dataset.__getitem__, 
            # but DINO needs multi-crop logic. 
            # Since our Dataset returns raw tensors, we apply the dict-returning transform here.
            # Ideally, move 'transform' into the Dataset class, but doing it here allows 
            # easy visualization of raw data if needed.
            
            # Since DataLoader collates inputs, we need to apply transform to each item
            # Use a custom collate_fn or apply loop here. 
            # For efficiency, let's assume the transform is inside the Dataset (Update below).
            pass 
            
            # Correction: Let's adjust the logic. The Dataset should return the dictionary of crops.
            # I will assume we wrapped the dataset with the transform properly.
            
            # STRUCTURE IF TRANSFORM IS IN DATASET:
            # batch_imgs is a list of lists of tensors? No, default collate stacks them.
            # DINO transforms return a dict of lists. Custom collate needed.
            # Let's handle the loop inputs carefully:
            
            # 1. Unpack data
            # Assuming custom collate was used or manual list handling
            global_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['global_crops']]
            local_crops = [c.to(self.device, non_blocking=True) for c in batch_imgs['local_crops']]
            
            # 2. Teacher Forward (Global Crops only)
            with torch.no_grad():
                # teacher_patches_list is [Tensor(Batch*2, 196, 768)]
                teacher_output, teacher_patches_list = self.teacher(global_crops) 
            
            # 3. Student Forward (All Crops)
            # student_patches_list is [Tensor(Global), Tensor(Local)]
            all_crops = global_crops + local_crops
            student_output, student_patches_list = self.student(all_crops)
            
            # 4. Calculate Losses
            
            # A. DINO Loss (CLS token matching)
            # Student output contains all crops. Teacher only global.
            loss_dino = self.dino_loss_fn(student_output, teacher_output)
            
            # B. KoLeo Loss (Student Batch Uniformity)
            # Only apply to global views of student to save compute
            n_global = len(global_crops)
            # student_output is concatenated (Batch * (2+8), Dim). 
            # Split to get global parts
            student_out_chunked = student_output.chunk(len(all_crops))
            student_global_cls = torch.cat(student_out_chunked[:n_global])
            loss_koleo = self.koleo_loss_fn(student_global_cls)
            
            # C. Gram Loss (Patch Texture Matching)
            # Only compute between Student Global and Teacher Global
            # (Comparing 96x96 local crops to 224x224 global crops via Gram matrix is 
            # mathematically messy due to different N_patches. We stick to global-global).
            
            # student_patches comes out as (Total_Batch, N_patches, Dim).
            # But wait, local crops have different N_patches than global crops!
            # The MultiCropWrapper will fail to stack 'student_patches' if shapes differ.
            # We need to rely on MultiCropWrapper handling lists or splitting outputs.
            # *Correction*: In the Modeling step, MultiCropWrapper returns patches only if shapes match
            # or we must modify it to return a list.
            # Assuming we extract patches for global views specifically:
            # loss_gram = self.gram_loss_fn(student_patches[:len(global_crops)*self.batch_size], teacher_patches)
            loss_gram = self.gram_loss_fn(student_patches_list[0], teacher_patches_list[0])

            # 5. Optimization
            loss = (self.w_dino * loss_dino) + (self.w_gram * loss_gram) + (self.w_koleo * loss_koleo)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 6. Update Teacher
            self.update_teacher_ema()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch_index} [{i}/{len(self.loader)}] "
                      f"Loss: {loss.item():.4f} (D:{loss_dino:.3f} G:{loss_gram:.3f} K:{loss_koleo:.3f})")

    def run(self):
        logger.info("Starting training...")
        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            self.train_one_epoch(epoch)
            # Save checkpoint logic here
            torch.save({
                'epoch': epoch,
                'student': self.student.state_dict(),
                'teacher': self.teacher.state_dict(),
            }, f"weights/checkpoint_{epoch}.pth")

# Custom Collate to handle the dictionary of lists from SonarDataTransform
def dino_collate_fn(batch):
    # batch is a list of dicts: [{'global_crops': [t1, t2], 'local_crops': [t3...]}, ...]
    output = {'global_crops': [], 'local_crops': []}
    
    # We want to stack: output['global_crops'] = [Batch_Crop1, Batch_Crop2]
    n_global = len(batch[0]['global_crops'])
    n_local = len(batch[0]['local_crops'])
    
    for i in range(n_global):
        output['global_crops'].append(torch.stack([item['global_crops'][i] for item in batch]))
        
    for i in range(n_local):
        output['local_crops'].append(torch.stack([item['local_crops'][i] for item in batch]))
        
    return output

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename=f"logs/{time.time()}.log",
        level=logging.INFO
    )

    trainer = Trainer()
    # Monkey patch the loader with the correct collate_fn and transform wrapper
    # (Since I simplified the Dataset class earlier, we apply transform inside dataset)
    
    # WRAPPING LOGIC:
    original_dataset = trainer.dataset
    transform_pipeline = trainer.transform
    
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, ds, tf):
            self.ds = ds
            self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            img = self.ds[idx]
            return self.tf(img)
            
    trainer.loader = DataLoader(
        TransformedDataset(original_dataset, transform_pipeline),
        batch_size=trainer.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=dino_collate_fn,
        drop_last=True
    )
    
    trainer.run()