import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import time
from datetime import datetime

# Define the character set
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_CHARS = len(CHARACTERS)
NUM_DIGITS = 10  # 0-9

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_files = list(self.root_dir.glob('*.jpg'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Extract label from filename
        label = img_path.stem
        
        # Validate label length
        if len(label) not in [6, 7]:
            raise ValueError(f"Invalid label length in {img_path.name}. Expected 6 or 7 characters, got {len(label)}")
        
        # For 6-character captchas: positions 1,4 are numbers
        # For 7-character captchas: positions 1,4,7 are numbers
        number_positions = [label[0], label[3]]  # positions 1,4 are always numbers
        if len(label) == 7:
            number_positions.append(label[6])  # position 7 is a number in 7-char captchas
        
        # Other positions are all non-number positions
        other_positions = []
        for i, char in enumerate(label):
            if i not in [0, 3, 6]:  # Skip number positions
                other_positions.append(char)
        
        # Convert characters to indices and pad to maximum length
        number_indices = [CHARACTERS.index(c) for c in number_positions]
        other_indices = [CHARACTERS.index(c) for c in other_positions]
        
        # Create masks for valid positions
        number_mask = torch.ones(3, dtype=torch.bool)  # 3 is max number positions
        other_mask = torch.ones(4, dtype=torch.bool)   # 4 is max other positions
        
        # Set mask to False for padded positions
        number_mask[len(number_indices):] = False
        other_mask[len(other_indices):] = False
        
        # Pad with zeros (which will be ignored by the mask)
        while len(number_indices) < 3:
            number_indices.append(0)
        while len(other_indices) < 4:
            other_indices.append(0)
        
        number_targets = torch.tensor(number_indices)
        other_targets = torch.tensor(other_indices)
        
        return img, (number_targets, other_targets, number_mask, other_mask)

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Separate heads for number positions and other positions
        self.number_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3 * NUM_DIGITS)  # Max 3 positions, each with 10 possible digits
        )
        
        self.other_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4 * NUM_CHARS)  # Max 4 positions, each with 36 possible characters
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Get predictions for both types of positions
        number_logits = self.number_head(features)
        other_logits = self.other_head(features)
        
        # Reshape logits to separate positions
        number_logits = number_logits.view(-1, 3, NUM_DIGITS)  # Always predict 3 positions, but some may be unused
        other_logits = other_logits.view(-1, 4, NUM_CHARS)    # Always predict 4 positions, but some may be unused
        
        return number_logits, other_logits

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, (number_targets, other_targets, number_mask, other_mask) in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            number_targets = number_targets.to(device)
            other_targets = other_targets.to(device)
            number_mask = number_mask.to(device)
            other_mask = other_mask.to(device)
            
            optimizer.zero_grad()
            
            number_logits, other_logits = model(images)
            
            # Calculate loss for both types of positions, using masks to ignore padded positions
            number_loss = criterion(
                number_logits.view(-1, NUM_DIGITS)[number_mask.view(-1)],
                number_targets.view(-1)[number_mask.view(-1)]
            )
            
            other_loss = criterion(
                other_logits.view(-1, NUM_CHARS)[other_mask.view(-1)],
                other_targets.view(-1)[other_mask.view(-1)]
            )
            
            # Combined loss
            loss = number_loss + other_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_number_preds = []
        all_number_targets = []
        all_other_preds = []
        all_other_targets = []
        
        with torch.no_grad():
            for images, (number_targets, other_targets, number_mask, other_mask) in val_loader:
                images = images.to(device)
                number_targets = number_targets.to(device)
                other_targets = other_targets.to(device)
                number_mask = number_mask.to(device)
                other_mask = other_mask.to(device)
                
                number_logits, other_logits = model(images)
                
                # Calculate validation loss
                number_loss = criterion(number_logits.view(-1, NUM_DIGITS)[number_mask.view(-1)], number_targets.view(-1)[number_mask.view(-1)])
                other_loss = criterion(other_logits.view(-1, NUM_CHARS)[other_mask.view(-1)], other_targets.view(-1)[other_mask.view(-1)])
                val_loss += (number_loss + other_loss).item()
                
                # Get predictions
                number_preds = torch.argmax(number_logits, dim=2)
                other_preds = torch.argmax(other_logits, dim=2)
                
                all_number_preds.extend(number_preds.cpu().numpy())
                all_number_targets.extend(number_targets.cpu().numpy())
                all_other_preds.extend(other_preds.cpu().numpy())
                all_other_targets.extend(other_targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        number_accuracy = accuracy_score(
            np.array(all_number_targets).flatten(),
            np.array(all_number_preds).flatten()
        )
        other_accuracy = accuracy_score(
            np.array(all_other_targets).flatten(),
            np.array(all_other_preds).flatten()
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Number Position Accuracy: {number_accuracy:.4f}')
        print(f'Other Position Accuracy: {other_accuracy:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print('Saved new best model')
    
    return train_losses, val_losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'model_checkpoints/model_positional_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CaptchaDataset('data/train2', transform=transform)
    val_dataset = CaptchaDataset('data/val2', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptchaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    num_epochs = 50
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, save_dir
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir / 'training_curves.png')
    plt.close()
    
    # Save training configuration
    config = {
        'model_architecture': str(model),
        'optimizer': str(optimizer),
        'criterion': str(criterion),
        'num_epochs': num_epochs,
        'batch_size': 32,
        'learning_rate': 0.001,
        'transform': str(transform),
        'timestamp': timestamp,
        'description': 'Model with separate heads for number positions (1,4,7) and other positions'
    }
    
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    main() 