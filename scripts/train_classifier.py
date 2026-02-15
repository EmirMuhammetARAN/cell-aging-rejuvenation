import os
import sys
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.models.classifier import Classifier
from torch.utils.data import random_split  

torch.backends.cudnn.benchmark = True         
torch.backends.cuda.matmul.allow_tf32 = True   
torch.backends.cudnn.allow_tf32 = True 
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_PATH = os.path.join(PARENT_DIR, 'data/processed/train')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 20
    BATCH_SIZE = 64


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=TRAIN_PATH, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True,persistent_workers=True)

    model = Classifier(output_size=2)
    model.to(DEVICE,memory_format=torch.channels_last)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss_total, train_correct, train_total = 0.0, 0, 0
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        model.train()

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch')
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0   
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE):
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        accuracy = val_correct / val_total if val_total > 0 else 0.0
        print(f'Accuracy: {train_correct/train_total:.4f} | Loss: {train_loss_total/len(train_loader):.4f} | Val Accuracy: {accuracy:.4f} | Val Loss: {val_loss_total/len(val_loader):.4f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(PARENT_DIR, 'checkpoints/classifier.pth'))
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')