import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor")

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, WeightedRandomSampler

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from models.ldm.model_diffusers import CellLDM

from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 64

EXPERIMENT_NAME = "v6_tightcrop"

if __name__ == "__main__":

    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR_LORA = 1e-4
    LR_EMBED = 5e-5
    GRADIENT_ACCUMULATION_STEPS = 4
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    import gc, glob

    model = CellLDM(num_classes=2)
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema() 

    optimizer = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.unet.parameters()), 'lr': LR_LORA},
        {'params': model.class_embed.parameters(), 'lr': LR_EMBED},
    ], weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    CHECKPOINT_PATH = "/mnt/windows/checpotint/checkpoints/ldm/checkpoint_v6_tightcrop_epoch_100.pt"
    OUTPUT_DIR = "/mnt/windows/checpotint/checkpoints/ldm"  # ← bu zaten tam path

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # ← sadece OUTPUT_DIR, çift yapma
    os.makedirs(os.path.join(OUTPUT_DIR, 'results', 'generated', 'ldm', EXPERIMENT_NAME), exist_ok=True)

    best_val_loss = float('inf')
    start_epoch = 0

    # Auto-resume
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_*.pt'))
    # ↑ ayrı 'checkpoints/ldm' ekleme
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[-1].split('.pt')[0]))
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'])
        if 'ema_unet_state_dict' in checkpoint:
            model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
        model.class_embed.load_state_dict(checkpoint['class_embed_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        del checkpoint
        gc.collect()
        
        best_model_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'ldm', f'best_model_{EXPERIMENT_NAME}.pt')
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            best_val_loss = best_checkpoint.get('val_loss', float('inf'))
            del best_checkpoint
            gc.collect()
            print(f"Resumed best validation loss: {best_val_loss:.4f}")
            
        print(f"Resuming training from epoch {start_epoch + 1}")

    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=transform, data_version='processed_v6')
    val_dataset = LDMDataset(root_dir=root_dir, split='test', transform=transform, data_version='processed_v6')

    # Balanced sampling: young/senescent eşit
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
                torch.nn.utils.clip_grad_norm_(model.class_embed.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                model.update_ema()  

            train_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1
            pbar.set_postfix(loss=f"{train_loss_sum/train_steps:.4f}")

        if accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(model.class_embed.parameters(), MAX_GRAD_NORM)
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
            use_ema = (epoch + 1) >= 5
            young_labels = torch.zeros(4, dtype=torch.long, device=DEVICE)
            senescent_labels = torch.ones(4, dtype=torch.long, device=DEVICE)

            torch.cuda.empty_cache()
            young_samples = model.sample(num_samples=4, device=DEVICE, labels=young_labels, use_ema=use_ema, guidance_scale=3.0).cpu()
            torch.cuda.empty_cache()
            senescent_samples = model.sample(num_samples=4, device=DEVICE, labels=senescent_labels, use_ema=use_ema, guidance_scale=3.0).cpu()
            torch.cuda.empty_cache()

            save_dir = os.path.join(OUTPUT_DIR, 'results', 'generated', 'ldm', EXPERIMENT_NAME)
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
                    translated_senes = model.translate(young_img, target_senes, strength=0.6, use_ema=use_ema, guidance_scale=3.0).cpu()
                    torch.cuda.empty_cache()

                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    translated_young = model.translate(senescent_img, target_young, strength=0.6, use_ema=use_ema, guidance_scale=3.0).cpu()
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
                    'class_embed_state_dict': model.class_embed.state_dict(),
                    'val_loss': avg_val_loss,
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(OUTPUT_DIR, 'checkpoints', 'ldm', f'best_model_{EXPERIMENT_NAME}.pt'))
                print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'ldm', f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_embed_state_dict': model.class_embed.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, ckpt_path)
            # Keep only last 1 checkpoint
            all_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'checkpoints', 'ldm', f'checkpoint_{EXPERIMENT_NAME}_epoch_*.pt')),
                               key=lambda x: int(x.split('_epoch_')[-1].split('.pt')[0]))
            for old_ckpt in all_ckpts[:-1]:
                os.remove(old_ckpt)
                print(f"  🗑 Eski checkpoint silindi: {os.path.basename(old_ckpt)}")

    print("Eğitim tamamlandı!")
