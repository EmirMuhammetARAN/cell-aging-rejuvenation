import os
import sys
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import tqdm

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.classifier.classifier import Classifier
from torch.utils.data import random_split  

torch.backends.cudnn.benchmark = True         
torch.backends.cuda.matmul.allow_tf32 = True   
torch.backends.cudnn.allow_tf32 = True 
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    PARENT_DIR = root_dir
    TRAIN_PATH = os.path.join(PARENT_DIR, 'data/processed_v4/train')
    VAL_PATH = os.path.join(PARENT_DIR, 'data/processed_v4/test')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 20
    BATCH_SIZE = 32


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=TRAIN_PATH, transform=transform)
    val_dataset = ImageFolder(root=VAL_PATH, transform=val_transform)

    # Balanced Sampler Logic
    targets = dataset.targets
    class_count = [0, 0]
    for t in targets: class_count[t] += 1
    class_weights = [1.0/class_count[0], 1.0/class_count[1]]
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = Classifier(output_size=2)
    model.to(DEVICE,memory_format=torch.channels_last)

    # START FRESH (NO RESUME TO CLEAR BIAS)
    print("Starting FRESH (No Resume) to eliminate class bias.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    best_accuracy = 0.0
    
    # Check baseline before starting
    model.eval()
    temp_correct, temp_total = 0, 0
    try:
        with torch.no_grad():
            for images, labels in tqdm.tqdm(val_loader, desc="Checking Baseline"):
                images, labels = images.to(DEVICE, memory_format=torch.channels_last), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                temp_total += labels.size(0)
                temp_correct += (predicted == labels).sum().item()
        best_accuracy = temp_correct / temp_total if temp_total > 0 else 0.0
        print(f"Starting FRESH Training with Baseline Accuracy: {best_accuracy:.4f}")
    except Exception as e:
        print(f"Baseline check failed, starting from 0.0: {e}")
        best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss_total, train_correct, train_total = 0.0, 0, 0
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        model.train()

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch')
        for images, labels in pbar:
            images, labels = images.to(DEVICE, memory_format=torch.channels_last), labels.to(DEVICE)
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
                    images, labels = images.to(DEVICE, memory_format=torch.channels_last), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        accuracy = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(accuracy)
        print(f'Accuracy: {train_correct/train_total:.4f} | Loss: {train_loss_total/len(train_loader):.4f} | Val Accuracy: {accuracy:.4f} | Val Loss: {val_loss_total/len(val_loader):.4f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = os.path.join(PARENT_DIR, 'checkpoints/classifier/classifier_v2.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')