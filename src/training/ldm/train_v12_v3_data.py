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

# ===== EXPERIMENT CONFIG =====
EXPERIMENT_NAME = "v12_v4_data"
# =============================
if __name__ == "__main__":

    BATCH_SIZE = 4  
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 250
    WARMUP_EPOCHS = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 2e-4
    GRADIENT_ACCUMULATION_STEPS = 4  
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5
    CHECKPOINT_EPOCHS = (50, 100, 150, 200, 250)
    LPIPS_WEIGHT = 0.0 

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

    # En son checkpoint'i bul
    import glob
    import re
    
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    latest_checkpoint = None
    max_epoch = -1
    
    for file in checkpoint_files:
        match = re.search(r'epoch_(\d+)\.pt', file)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_checkpoint = file

    best_model_path = os.path.join(CHECKPOINT_DIR, f'best_model_{EXPERIMENT_NAME}.pt')
    
    # Eğer en son checkpoint varsa onu, yoksa best_model'i kullan
    resume_path = latest_checkpoint if latest_checkpoint else best_model_path

    if os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
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
        else:
            # En son epoch checkpoint'ten dönüldüğünde val_loss içinde olmayabilir.
            # Best model'in üzerine yazmamak için best_model dosyasından val_loss'u okuyalım.
            if os.path.exists(best_model_path):
                try:
                    best_ckpt = torch.load(best_model_path, map_location='cpu')
                    if 'val_loss' in best_ckpt:
                        best_val_loss = best_ckpt['val_loss']
                    del best_ckpt
                except Exception:
                    pass
                    
        del checkpoint
        gc.collect()
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    else:
        print("Starting from scratch (no checkpoint found for v10_custom_vae_lpips)")

    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=train_transform, data_version='processed_v4')
    val_dataset = LDMDataset(root_dir=root_dir, split='test', transform=val_transform, data_version='processed_v4')

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

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                loss = model(images, labels=labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            accumulation_counter += 1

            if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                model.update_ema()  

            train_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1
            pbar.set_postfix(loss=f"{train_loss_sum/train_steps:.4f}")

        if accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            model.update_ema()

        lr_scheduler.step()
        avg_train_loss = train_loss_sum / train_steps

        if (epoch + 1) % VAL_FREQ == 0:
            model.unet.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                    images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                    labels = labels.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                        loss = model(images, labels=labels)
                    val_loss_sum += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / val_steps
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            model.unet.eval()
            use_ema = (epoch + 1) >= 10
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
                    # 3. DEĞİŞİKLİK: AGING ŞAMPİYON AYARLARI (strength 0.70, cfg 4.0)
                    translated_senes = model.translate(young_img, target_senes, strength=0.70, use_ema=use_ema, guidance_scale=4.0).cpu()
                    torch.cuda.empty_cache()

                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    # 3. DEĞİŞİKLİK: REJUVENATION ŞAMPİYON AYARLARI (strength 0.65, cfg 3.0)
                    translated_young = model.translate(senescent_img, target_young, strength=0.65, use_ema=use_ema, guidance_scale=3.0).cpu()
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
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Checkpoint kaydedildi: epoch_{epoch+1}.pt")

    print("Eğitim tamamlandı!")