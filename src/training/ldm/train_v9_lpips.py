import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4294967296"

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, WeightedRandomSampler

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from models.ldm.model_lpips import CellLDM

from tqdm import tqdm
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 64

EXPERIMENT_NAME = "v9_base_fixed_v2" # Same experiment name but new Phase
PHASE_NAME = "v9_lpips_v2"

if __name__ == "__main__":

    BATCH_SIZE = 4  
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 100 # More time with lower LR
    WARMUP_EPOCHS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 2e-5 # Very low LR for stable perceptual refinement
    GRADIENT_ACCUMULATION_STEPS = 4  
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 10
    CHECKPOINT_EPOCHS = (10, 25, 50, 75) 
    LPIPS_WEIGHT = 0.5 # Start with 0.5 to balance with MSE

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    vae_path = os.path.join(root_dir, 'checkpoints', 'vae_finetuned', 'best')
    model = CellLDM(lpips_weight=LPIPS_WEIGHT, vae_path=vae_path) 
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema() 

    optimizer = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.unet.parameters()), 'lr': LR},
    ], weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    OUTPUT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', PHASE_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # LOAD BASE CHECKPOINT
    resume_path = os.path.join(OUTPUT_DIR, f'best_model_{EXPERIMENT_NAME}.pt')
    if os.path.exists(resume_path):
        print(f"Loading Base Model for LPIPS Fine-tuning: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
        if 'ema_unet_state_dict' in checkpoint:
            model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
        print(f"Starting Phase B from best val_loss: {checkpoint['val_loss']:.4f}")
        best_val_loss = float('inf') # Reset for new PHASE
        start_epoch = 0
        del checkpoint
        gc.collect()
    else:
        print("ERROR: Base model checkpoint not found for Phase B!")
        sys.exit(1)

    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=train_transform, data_version='processed_v2')
    val_dataset = LDMDataset(root_dir=root_dir, split='test', transform=val_transform, data_version='processed_v2')

    young_count = train_dataset.labels.count(0)
    senes_count = train_dataset.labels.count(1)
    class_weights = {0: 1.0 / young_count, 1: 1.0 / senes_count}
    sample_weights = [class_weights[l] for l in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.unet.train()
        train_loss_sum = 0.0
        train_steps = 0
        optimizer.zero_grad()
        accumulation_counter = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Phase B-LPIPS]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                loss = model(images, labels) / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            
            train_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1
            accumulation_counter += 1

            if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                model.update_ema()
                accumulation_counter = 0

            pbar.set_postfix({'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        lr_scheduler.step()
        avg_train_loss = train_loss_sum / train_steps
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % VAL_FREQ == 0:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for v_images, v_labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                    v_images = v_images.to(DEVICE)
                    v_labels = v_labels.to(DEVICE)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                        v_loss = model(v_images, v_labels)
                    val_loss_sum += v_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_sum / val_steps
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save sample translations
            example_images = next(iter(val_dataloader))[0][:1].to(DEVICE)
            example_labels = torch.tensor([1], device=DEVICE) # Challenge: Turn to Senescent
            translated = model.translate(example_images, example_labels, strength=0.75)
            save_image(translated, os.path.join(RESULTS_DIR, f"epoch_{epoch+1}_translation.png"))
            print(f"  ✓ Translation examples saved")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(OUTPUT_DIR, f'best_model_{PHASE_NAME}.pt')
                torch.save({
                    'epoch': epoch,
                    'unet_state_dict': model.unet.state_dict(),
                    'ema_unet_state_dict': model.ema_unet.state_dict() if model.ema_unet else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, save_path)
                print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            torch.save(model.unet.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoint_unet_{PHASE_NAME}_epoch_{epoch+1}.pt'))
            print(f"  ✓ Checkpoint saved: epoch_{epoch+1}.pt")

    print("Phase B LPIPS Fine-tuning Complete!")
