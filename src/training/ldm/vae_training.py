import os, sys, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from tqdm import tqdm
import torch.nn.functional as F

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.ldm.ldm_dataset import LDMDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = 'cuda'
LR_ENCODER = 5e-6  
LR_DECODER = 1e-5  
EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUM = 8
OUTPUT_DIR = '/mnt/windows/checpotint/vae_finetuned'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visuals'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    dataset = LDMDataset(
        root_dir=root_dir, 
        split='train', 
        transform=transform, 
        data_version='processed_v6'
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(DEVICE)

    vae.encoder.requires_grad_(True)
    vae.decoder.requires_grad_(True)
    vae.quant_conv.requires_grad_(True)
    vae.post_quant_conv.requires_grad_(True)

    optimizer = torch.optim.AdamW([
        {'params': vae.encoder.parameters(),         'lr': LR_ENCODER},
        {'params': vae.quant_conv.parameters(),      'lr': LR_ENCODER},
        {'params': vae.decoder.parameters(),         'lr': LR_DECODER},
        {'params': vae.post_quant_conv.parameters(), 'lr': LR_DECODER},
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')

    # Görsel karşılaştırma için sabit 4 örnek seç (her epoch aynı)
    fixed_indices = [0, len(dataset)//4, len(dataset)//2, len(dataset)*3//4]
    fixed_imgs = torch.stack([dataset[i][0] for i in fixed_indices]).to(DEVICE)

    for epoch in range(EPOCHS):
        vae.train()
        total_recon = 0
        total_kl = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{EPOCHS}")
        
        for step, (images, _) in enumerate(pbar):
            images = images.to(DEVICE, memory_format=torch.channels_last)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                posterior = vae.encode(images).latent_dist
                z = posterior.sample()
                recon = vae.decode(z).sample
                
                recon_loss = F.l1_loss(recon, images)
                kl_loss = posterior.kl().mean() * 1e-6
                loss = (recon_loss + kl_loss) / GRAD_ACCUM
            
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            pbar.set_postfix(
                recon=f"{total_recon/(step+1):.4f}",
                kl=f"{total_kl/(step+1):.6f}"
            )
        
        avg_loss = (total_recon + total_kl) / len(dataloader)
        print(f"Epoch {epoch+1} | Recon: {total_recon/len(dataloader):.4f} | "
              f"KL: {total_kl/len(dataloader):.6f}")
        
        if (epoch + 1) % 5 == 0:
            vae.save_pretrained(os.path.join(OUTPUT_DIR, f'epoch_{epoch+1}'))
            print(f"  ✓ Snapshot: epoch_{epoch+1}")

            # Görsel kontrol
            vae.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z = vae.encode(fixed_imgs).latent_dist.sample()
                recons = vae.decode(z).sample

            # orijinal ve recon yan yana: [orig1, recon1, orig2, recon2, ...]
            grid_imgs = []
            for i in range(len(fixed_indices)):
                orig  = (fixed_imgs[i].float().cpu()  * 0.5 + 0.5).clamp(0, 1)
                recon = (recons[i].float().cpu() * 0.5 + 0.5).clamp(0, 1)
                grid_imgs.extend([orig, recon])

            save_image(
                torch.stack(grid_imgs),
                os.path.join(OUTPUT_DIR, 'visuals', f'epoch_{epoch+1}.png'),
                nrow=2      # her satırda: orijinal | recon
            )
            print(f"  ✓ Görsel: visuals/epoch_{epoch+1}.png")
            vae.train()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            vae.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            print(f"  ✓ Best VAE (loss: {best_loss:.4f})")

    vae.save_pretrained(os.path.join(OUTPUT_DIR, 'final'))
    print("VAE fine-tune tamamlandı!")