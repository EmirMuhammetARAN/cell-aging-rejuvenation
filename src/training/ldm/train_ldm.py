import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor")

import torch
from torchvision import transforms
from torchvision.utils import save_image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from models.ldm.model import CellLDM
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 64

if __name__ == "__main__":

    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 1
    NUM_EPOCHS = 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 2e-4
    GRADIENT_ACCUMULATION_STEPS = 4
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = LDMDataset(root_dir=root_dir, split='train', transform=transform)
    val_dataset = LDMDataset(root_dir=root_dir, split='test', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                   pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4,
                                pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = CellLDM(num_classes=2)
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema() 

    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=LR, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    os.makedirs(os.path.join(root_dir, 'checkpoints', 'ldm'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'results', 'generated', 'ldm'), exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
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

            young_samples = model.sample(num_samples=4, device=DEVICE, labels=young_labels, use_ema=use_ema)
            senescent_samples = model.sample(num_samples=4, device=DEVICE, labels=senescent_labels, use_ema=use_ema)
            
            save_path_young = os.path.join(root_dir, 'results', 'generated', 'ldm', f'epoch_{epoch+1}_young.png')
            save_path_senes = os.path.join(root_dir, 'results', 'generated', 'ldm', f'epoch_{epoch+1}_senescent.png')
            save_image(young_samples, save_path_young, nrow=4)
            save_image(senescent_samples, save_path_senes, nrow=4)

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
                    translated_senes = model.translate(young_img, target_senes, strength=0.6, use_ema=use_ema)
                    
                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    translated_young = model.translate(senescent_img, target_young, strength=0.6, use_ema=use_ema)
                    
                    young_orig_vis = (young_img.clamp(-1, 1) + 1) / 2
                    senes_orig_vis = (senescent_img.clamp(-1, 1) + 1) / 2
                    
                    y2s_grid = torch.cat([young_orig_vis, translated_senes], dim=0)
                    save_image(y2s_grid, os.path.join(root_dir, 'results', 'generated', 'ldm',
                               f'epoch_{epoch+1}_translate_young2senes.png'), nrow=2)
                    
                    s2y_grid = torch.cat([senes_orig_vis, translated_young], dim=0)
                    save_image(s2y_grid, os.path.join(root_dir, 'results', 'generated', 'ldm',
                               f'epoch_{epoch+1}_translate_senes2young.png'), nrow=2)
                    
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
                }, os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model.pt'))
                print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(root_dir, 'checkpoints', 'ldm', f'checkpoint_epoch_{epoch+1}.pt'))

    print("Eğitim tamamlandı!")
