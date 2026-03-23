import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4294967296"

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, WeightedRandomSampler

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from models.ldm.model_lpips import CellLDM
from models.ldm.latent_discriminator import LatentDiscriminator

from tqdm import tqdm
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 64

EXPERIMENT_NAME = "v7_gan"  

if __name__ == "__main__":

    BATCH_SIZE = 2
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 30
    WARMUP_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Learning rates
    LR_G = 1e-5     # Generator (UNet) LR 
    LR_D = 5e-5     # Discriminator LR
    
    GRADIENT_ACCUMULATION_STEPS = 8
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5
    CHECKPOINT_EPOCHS = (5, 10, 15, 20, 25, 30) 
    
    # Loss weights
    L1_WEIGHT = 0.03  # Latent Structural Loss
    ADV_WEIGHT = 0.05 # Adversarial Loss weight for Generator

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 1. Initialize Models
    model = CellLDM(lpips_weight=L1_WEIGHT) 
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    
    discriminator = LatentDiscriminator(in_channels=4, num_classes=2, ndf=64)
    discriminator.to(DEVICE, memory_format=torch.channels_last)

    # 2. Load the best V2 LPIPS model as the starting point for UNet
    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', EXPERIMENT_NAME)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_v2_lpips_ft.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading base UNet from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
        model.init_ema() 
        if 'ema_unet_state_dict' in checkpoint:
            model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'], strict=True)
        del checkpoint
        gc.collect()
    else:
        print(f"Error: Base checkpoint not found at {checkpoint_path}. Train the LPIPS model first.")
        sys.exit(1)

    # 3. Optimizers
    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.unet.parameters()), lr=LR_G, betas=(0.9, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999), weight_decay=1e-2)

    # Schedulers
    warmup_G = torch.optim.lr_scheduler.LinearLR(optimizer_G, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    scheduler_G = torch.optim.lr_scheduler.SequentialLR(optimizer_G, schedulers=[warmup_G, cosine_G], milestones=[WARMUP_EPOCHS])

    warmup_D = torch.optim.lr_scheduler.LinearLR(optimizer_D, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    scheduler_D = torch.optim.lr_scheduler.SequentialLR(optimizer_D, schedulers=[warmup_D, cosine_D], milestones=[WARMUP_EPOCHS])

    # 4. Dataset
    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=transform, data_version='processed_v2')
    val_dataset = LDMDataset(root_dir=root_dir, split='test', transform=transform, data_version='processed_v2')

    young_count = train_dataset.labels.count(0)
    senes_count = train_dataset.labels.count(1)
    class_weights = {0: 1.0 / young_count, 1: 1.0 / senes_count}
    sample_weights = [class_weights[l] for l in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss = float('inf')

    # GAN Loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        model.unet.train()
        discriminator.train()
        
        train_loss_G_sum = 0.0
        train_loss_D_sum = 0.0
        train_steps = 0
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        accumulation_counter = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                # Setup
                latents = model.encode(images)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, model.train_scheduler.config.num_train_timesteps, 
                                           (latents.shape[0],), device=DEVICE).long()
                noisy_latents = model.train_scheduler.add_noise(latents, noise, timesteps)
                
                # CFG Dropout
                drop_mask = torch.rand(labels.shape[0], device=DEVICE) < model.cfg_drop_prob
                unet_labels = labels.clone()
                unet_labels[drop_mask] = model.num_classes  

                # Forward UNet
                noise_pred = model.unet(noisy_latents, timesteps, class_labels=unet_labels).sample
                
                # Compute pred_x0 for L1 and Discriminator
                alpha_prod_t = model.train_scheduler.alphas_cumprod.to(DEVICE)[timesteps].view(-1, 1, 1, 1)
                pred_x0 = (noisy_latents - (1 - alpha_prod_t).sqrt() * noise_pred) / (alpha_prod_t.sqrt() + 1e-8)
                
                # -----------------------
                # 1. Train Discriminator
                # -----------------------
                # Real latents are positive (1), Fake pred_x0 are negative (0)
                # We use drop_mask to only condition discriminator if CFG wasn't dropped
                d_labels = labels.clone()
                
                real_logits = discriminator(latents, labels=d_labels)
                fake_logits = discriminator(pred_x0.detach(), labels=d_labels) # Detach to not backward through UNet
                
                d_loss_real = bce_loss(real_logits, torch.ones_like(real_logits))
                d_loss_fake = bce_loss(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (d_loss_real + d_loss_fake) / 2.0
                d_loss = d_loss / GRADIENT_ACCUMULATION_STEPS

            # Backward D
            d_loss.backward()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                # -----------------------
                # 2. Train Generator (UNet)
                # -----------------------
                # Generator wants fake_logits to be classified as real (1)
                mse_loss = F.mse_loss(noise_pred, noise)
                latent_l1 = F.l1_loss(pred_x0, latents)
                
                gen_fake_logits = discriminator(pred_x0, labels=d_labels) # No detach!
                adv_loss = bce_loss(gen_fake_logits, torch.ones_like(gen_fake_logits))
                
                g_loss = mse_loss + (L1_WEIGHT * latent_l1) + (ADV_WEIGHT * adv_loss)
                g_loss = g_loss / GRADIENT_ACCUMULATION_STEPS

            # Backward G
            g_loss.backward()
            
            accumulation_counter += 1

            if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), MAX_GRAD_NORM)
                optimizer_D.step()
                optimizer_D.zero_grad()
                
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer_G.step()
                optimizer_G.zero_grad()
                
                model.update_ema()  

            train_loss_G_sum += g_loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_loss_D_sum += d_loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_steps += 1
            
            pbar.set_postfix(G=f"{train_loss_G_sum/train_steps:.4f}", D=f"{train_loss_D_sum/train_steps:.4f}")

        if accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), MAX_GRAD_NORM)
            optimizer_D.step()
            optimizer_D.zero_grad()
            
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
            optimizer_G.step()
            optimizer_G.zero_grad()
            model.update_ema()

        scheduler_G.step()
        scheduler_D.step()
        
        avg_train_loss_G = train_loss_G_sum / train_steps
        avg_train_loss_D = train_loss_D_sum / train_steps

        # --- Validation & Sampling Phase ---
        if (epoch + 1) % VAL_FREQ == 0:
            model.unet.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                    images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                    labels = labels.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                        loss = model(images, labels=labels) # Just checks standard forward MSE/L1 loss
                    val_loss_sum += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / val_steps
            print(f"Epoch {epoch+1} | G Loss: {avg_train_loss_G:.4f} | D Loss: {avg_train_loss_D:.4f} | Val G Loss: {avg_val_loss:.4f}")

            use_ema = (epoch + 1) >= 5
            young_labels = torch.zeros(4, dtype=torch.long, device=DEVICE)
            senescent_labels = torch.ones(4, dtype=torch.long, device=DEVICE)

            # Sample Generation 
            torch.cuda.empty_cache()
            young_samples = model.sample(num_samples=4, device=DEVICE, labels=young_labels, use_ema=use_ema, guidance_scale=2.0).cpu()
            torch.cuda.empty_cache()
            senescent_samples = model.sample(num_samples=4, device=DEVICE, labels=senescent_labels, use_ema=use_ema, guidance_scale=2.0).cpu()
            torch.cuda.empty_cache()

            save_image(young_samples, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_young.png'), nrow=4)
            save_image(senescent_samples, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_senescent.png'), nrow=4)

            # Translation Generation
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
                    translated_senes = model.translate(young_img, target_senes, strength=0.85, use_ema=use_ema, guidance_scale=2.5).cpu()
                    torch.cuda.empty_cache()

                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    translated_young = model.translate(senescent_img, target_young, strength=0.75, use_ema=use_ema, guidance_scale=2.0).cpu()
                    torch.cuda.empty_cache()

                    young_orig_vis = (young_img.clamp(-1, 1) + 1) / 2
                    senes_orig_vis = (senescent_img.clamp(-1, 1) + 1) / 2

                    y2s_grid = torch.cat([young_orig_vis.cpu(), translated_senes], dim=0)
                    save_image(y2s_grid, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_translate_young2senes.png'), nrow=2)

                    s2y_grid = torch.cat([senes_orig_vis.cpu(), translated_young], dim=0)
                    save_image(s2y_grid, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_translate_senes2young.png'), nrow=2)

                    del young_img, senescent_img, translated_senes, translated_young
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ⚠ Translation örneği oluşturulamadı: {e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'unet_state_dict': model.unet.state_dict(),
                    'ema_unet_state_dict': model.ema_unet.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(CHECKPOINT_DIR, f'best_model_{EXPERIMENT_NAME}.pt'))
                print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1} | G Loss: {avg_train_loss_G:.4f} | D Loss: {avg_train_loss_D:.4f}")

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
            }, ckpt_path)

    print("Eğitim tamamlandı!")
