# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from Res18_model import ResNet18_CIFAR 

# ====================== This guard is REQUIRED on Windows ======================
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize the base model
    model = ResNet18_CIFAR(num_classes=10).to(device)
    
    # 2. Check for saved weights and load them BEFORE compiling
    model_path = "resnet18_best.pth"
    if os.path.exists(model_path):
        print(f" Found saved model '{model_path}'. Loading weights...")
        # weights_only=True is a security best practice for loading PyTorch weights
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Strip the '_orig_mod.' prefix that torch.compile adds to saved weights
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state_dict)
        print(" Weights loaded successfully!")
    else:
        print(" No saved model found. Training from scratch...")

    # 3. Compile the model
    model = torch.compile(model, backend="aot_eager")

    # ====================== DATA ======================
    data_root = r'E:\AI\res'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    testset  = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)

    # Reduced num_workers + persistent_workers for stability on laptop
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    testloader  = DataLoader(testset,  batch_size=512, shuffle=False, 
                             num_workers=4, pin_memory=True)

    print(f"Loaded {len(trainset)} training images and {len(testset)} test images")

    # ====================== LOSS + OPTIMIZER ======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')                    # Fixed instantiation
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # MixUp
    def mixup_data(x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    # ====================== TRAINING LOOP ======================
    epochs = 10
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                use_mixup = True
            else:
                use_mixup = False
            
            optimizer.zero_grad()
            
            # Fixed autocast syntax
            with autocast('cuda'):
                outputs = model(inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{train_loss/len(trainloader):.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} - Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"   >>> New best: {best_acc:.2f}% saved!")

    print(f"\n Training finished! Best accuracy: {best_acc:.2f}%")
