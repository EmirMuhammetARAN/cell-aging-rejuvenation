import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from src.data.dataset_maker import DatasetMaker
from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.cyclegan_gan_model import CycleGANModel
from tqdm import tqdm
from PIL import Image


torch.backends.cudnn.benchmark = True         
torch.backends.cuda.matmul.allow_tf32 = True   
torch.backends.cudnn.allow_tf32 = True 
torch.set_float32_matmul_precision('medium')



if __name__ == "__main__":
    # Configuration
    YOUNG_PATH = os.path.join(root_dir, 'data', 'processed','train','young')
    SENESCENT_PATH = os.path.join(root_dir, 'data', 'processed', 'train', 'senescent')
    BATCH_SIZE = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 100
    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints')
    START_EPOCH = 0
    RESUME_EPOCH = 50
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, f'cyclegan_v1_epoch_{RESUME_EPOCH}.pth') if RESUME_EPOCH > 0 else None
    parent_dir = root_dir  # For compatibility with rest of code
    epoch_loss_G = []
    epoch_loss_D_A = []
    epoch_loss_D_B = []





    # Data loading and preprocessing
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = DatasetMaker(young_path=YOUNG_PATH, senescent_path=SENESCENT_PATH, transform=transform_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True)




    # Initialize model
    model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE)
    model.to(DEVICE,memory_format=torch.channels_last)


    best_loss_G = 2.56
    
    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print('Checkpoint loaded successfully.')
        START_EPOCH = RESUME_EPOCH
    else:
        print('No checkpoint found, starting from scratch.')

    model = torch.compile(model)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def lr_lambda(epoch):
        if epoch < NUM_EPOCHS // 2:
            return 1.0
        else:
            return 1.0 - (epoch - NUM_EPOCHS // 2) / (NUM_EPOCHS // 2)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, lr_lambda=lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D_A, lr_lambda=lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D_B, lr_lambda=lr_lambda)



    

    scaler = torch.amp.GradScaler('cuda')

    ## Training loop
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch in pbar:
            loss_G, loss_D_A, loss_D_B = model.train_step(batch,scaler)
            epoch_loss_G.append(loss_G.item())
            epoch_loss_D_A.append(loss_D_A.item())
            epoch_loss_D_B.append(loss_D_B.item())
            pbar.set_postfix({
                'Loss G': f'{loss_G:.4f}',
                'Loss D_A': f'{loss_D_A:.4f}',
                'Loss D_B': f'{loss_D_B:.4f}'
            })
        
        avg_loss_G = sum(epoch_loss_G)/len(epoch_loss_G)
        avg_loss_D_A = sum(epoch_loss_D_A)/len(epoch_loss_D_A)
        avg_loss_D_B = sum(epoch_loss_D_B)/len(epoch_loss_D_B)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss G: {avg_loss_G:.4f}, Average Loss D_A: {avg_loss_D_A:.4f}, Average Loss D_B: {avg_loss_D_B:.4f}')
        
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            torch.save(model._orig_mod.state_dict(), os.path.join(CHECKPOINT_DIR, 'cyclegan_best_v1.pth'))
            print(f'New best model saved with Loss G: {best_loss_G:.4f}')
        
        if (epoch + 1) % 10 == 0:
           torch.save(model._orig_mod.state_dict(), os.path.join(CHECKPOINT_DIR, f'cyclegan_v1_epoch_{epoch+1}.pth'))
           print(f'Checkpoint saved for epoch {epoch+1}.')
        
        epoch_loss_G = []
        epoch_loss_D_A = []
        epoch_loss_D_B = []
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_img = Image.open(os.path.join(parent_dir, 'data', 'processed', 'test', 'young', os.listdir(os.path.join(parent_dir, 'data', 'processed', 'test', 'young'))[0])).convert('RGB')
            test_tensor = transform_test(test_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_senescent = model._orig_mod.G_AB(test_tensor)
            fake_img = fake_senescent.squeeze(0).cpu()
            fake_img = fake_img * 0.5 + 0.5
            fake_img = fake_img.clamp(0, 1)
            save_image = transforms.ToPILImage()(fake_img)
            os.makedirs(os.path.join(parent_dir, 'results'), exist_ok=True)
            save_image.save(os.path.join(parent_dir, 'results', f'sample_epoch_{epoch+1}.png'))
            model.train()