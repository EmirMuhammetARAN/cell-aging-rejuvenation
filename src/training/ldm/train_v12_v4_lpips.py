import os, sys, re, glob
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4294967296"

import csv
import random
import time
import numpy as np
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

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


EXPERIMENT_NAME = "v12_v5_data_lpips"
BASE_CHECKPOINT = "best_model_v12_v5_data.pt"
if __name__ == "__main__":
    SEED = 2026
    set_seed(SEED)

    BATCH_SIZE = 4  
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 50  # LPIPS fine-tune icin kisa epoch yeterli
    WARMUP_EPOCHS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 5e-5  # Dusuk LR, fine-tune icin
    GRADIENT_ACCUMULATION_STEPS = 4  
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5
    MILESTONE_EPOCHS = (25, 50)
    SAVE_INTERVAL = 5
    LPIPS_WEIGHT = 0.1

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

    model = CellLDM(lpips_weight=LPIPS_WEIGHT) 

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
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', EXPERIMENT_NAME)
    CHECKPOINT_DIR = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_last.pt')
    base_checkpoint_path = os.path.join(CHECKPOINT_DIR, BASE_CHECKPOINT)

    resume_path = None
    if os.path.exists(last_ckpt_path):
        resume_path = last_ckpt_path
    else:
        lpips_checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_*.pt')
        lpips_checkpoint_files = glob.glob(lpips_checkpoint_pattern)
        max_epoch = -1
        for file in lpips_checkpoint_files:
            match = re.search(r'epoch_(\d+)\.pt', file)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    resume_path = file

    if resume_path:
        print(f"Resuming LPIPS fine-tune from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
        if 'ema_unet_state_dict' in checkpoint:
            model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'], strict=True)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        del checkpoint; gc.collect()
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    elif os.path.exists(base_checkpoint_path):
        print(f"Starting LPIPS fine-tune from base v5 model: {base_checkpoint_path}")
        checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
        if 'ema_unet_state_dict' in checkpoint:
            model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'], strict=True)
        # Optimizer ve scheduler sifirdan baslasin (yeni LR ile)
        del checkpoint; gc.collect()
        print(f"Base v5 model loaded. Starting LPIPS fine-tune from epoch 0.")
    else:
        print(f"ERROR: Base checkpoint not found at {base_checkpoint_path}")
        sys.exit(1)

    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=train_transform, data_version='processed_v5')
    val_dataset = LDMDataset(root_dir=root_dir, split='val', transform=val_transform, data_version='processed_v5')

    young_count = train_dataset.labels.count(0)
    senes_count = train_dataset.labels.count(1)
    class_weights = {0: 1.0 / young_count, 1: 1.0 / senes_count}
    sample_weights = [class_weights[l] for l in train_dataset.labels]
    sampler_gen = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=sampler_gen)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True,
                                  worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)

    log_path = os.path.join(RESULTS_DIR, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'epoch_time_s'])

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        model.unet.train()
        train_loss_sum = 0.0
        train_steps = 0
        optimizer.zero_grad()
        accumulation_counter = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train LPIPS]")
        for images, labels in pbar:
            images = images.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32):
                loss = model(images, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            accumulation_counter += 1

            if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                model.update_ema()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1
            pbar.set_postfix({
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        if accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            model.update_ema()
            optimizer.zero_grad()

        # Step LR scheduler once per epoch (matches WARMUP_EPOCHS=3, T_max=47)
        lr_scheduler.step()
        avg_train_loss = train_loss_sum / train_steps
        ep_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']
        avg_val_loss = float('nan')
        
        if (epoch + 1) % VAL_FREQ == 0:
            model.unet.eval()
            val_loss_sum = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for val_images, val_labels in val_dataloader:
                    val_images = val_images.to(DEVICE, memory_format=torch.channels_last)
                    val_labels = val_labels.to(DEVICE)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32):
                        loss = model(val_images, val_labels)
                        val_loss_sum += loss.item()
                        val_steps += 1

            avg_val_loss = val_loss_sum / val_steps
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {cur_lr:.2e} ({ep_time:.1f}s)")

            model.unet.eval()
            use_ema = True  # LPIPS fine-tune'da her zaman EMA kullan
            young_labels = torch.zeros(4, dtype=torch.long, device=DEVICE)
            senescent_labels = torch.ones(4, dtype=torch.long, device=DEVICE)

            torch.cuda.empty_cache()
            young_samples = model.sample(num_samples=4, device=DEVICE, labels=young_labels, use_ema=use_ema, guidance_scale=3.0, num_steps=50).cpu()
            torch.cuda.empty_cache()
            senescent_samples = model.sample(num_samples=4, device=DEVICE, labels=senescent_labels, use_ema=use_ema, guidance_scale=3.0, num_steps=50).cpu()
            torch.cuda.empty_cache()

            save_dir = RESULTS_DIR
            save_image(young_samples, os.path.join(save_dir, f'epoch_{epoch+1}_young.png'), nrow=4)
            save_image(senescent_samples, os.path.join(save_dir, f'epoch_{epoch+1}_senescent.png'), nrow=4)

            try:
                val_iter = iter(val_dataloader)
                young_img, senescent_img = None, None
                for val_img, val_label in val_iter:
                    if val_label.item() == 0 and young_img is None:
                        young_img = val_img.to(DEVICE, memory_format=torch.channels_last)
                    elif val_label.item() == 1 and senescent_img is None:
                        senescent_img = val_img.to(DEVICE, memory_format=torch.channels_last)
                    if young_img is not None and senescent_img is not None:
                        break

                if young_img is not None and senescent_img is not None:
                    target_senes = torch.ones(1, dtype=torch.long, device=DEVICE)
                    # Aging sampling params
                    translated_senes = model.translate(young_img, target_senes, strength=0.85, use_ema=use_ema, guidance_scale=6.0).cpu()
                    torch.cuda.empty_cache()

                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    # Rejuvenation sampling params
                    translated_young = model.translate(senescent_img, target_young, strength=0.75, use_ema=use_ema, guidance_scale=5.0).cpu()
                    torch.cuda.empty_cache()

                    young_orig_vis = (young_img.clamp(-1, 1) + 1) / 2
                    senes_orig_vis = (senescent_img.clamp(-1, 1) + 1) / 2

                    y2s_grid = torch.cat([young_orig_vis.cpu(), translated_senes], dim=0)
                    save_image(y2s_grid, os.path.join(save_dir, f'epoch_{epoch+1}_translate_young2senes.png'), nrow=2)

                    s2y_grid = torch.cat([senes_orig_vis.cpu(), translated_young], dim=0)
                    save_image(s2y_grid, os.path.join(save_dir, f'epoch_{epoch+1}_translate_senes2young.png'), nrow=2)

                    del young_img, senescent_img, translated_senes, translated_young
                    torch.cuda.empty_cache()

                    print(f"  ✓ Translation örnekleri kaydedildi")
            except Exception as e:
                print(f"  ⚠ Translation örneği oluşturulamadı: {e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'unet_state_dict': model.unet.state_dict(),
                    'ema_unet_state_dict': model.ema_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(CHECKPOINT_DIR, f'best_model_{EXPERIMENT_NAME}.pt'))
                print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | LR: {cur_lr:.2e} ({ep_time:.1f}s)")

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.5f}", f"{avg_val_loss:.5f}", f"{cur_lr:.3e}", f"{ep_time:.1f}"])

        if (epoch + 1) % SAVE_INTERVAL == 0:
            last_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_last.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, last_path)
            
        if (epoch + 1) in MILESTONE_EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Milestone Checkpoint kaydedildi: epoch_{epoch+1}.pt")

    print("LPIPS Fine-tune tamamlandı!")
