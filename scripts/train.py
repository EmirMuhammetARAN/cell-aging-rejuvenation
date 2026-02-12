import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.dataset_maker import DatasetMaker
from src.models.generator_resnet import GeneratorResNet
from src.models.discriminator import Discriminator
from src.models.cyclegan_gan_model import CycleGANModel

YOUNG_PATH = os.path.join(parent_dir, 'data', 'processed','train','young')
SENESCENT_PATH = os.path.join(parent_dir, 'data', 'processed', 'train', 'senescent')
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 50

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = DatasetMaker(young_path=YOUNG_PATH, senescent_path=SENESCENT_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE)
model.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(dataloader):
        loss_D_A, loss_D_B = model.train_step(batch)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss D_A: {loss_D_A}, Loss D_B: {loss_D_B}')