import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def training_epoch(model, train_loader, optimizer, criterion, device, unsqueezeY = False):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for X, Y in tqdm(train_loader, desc="Training", leave=False):
        #batch_size = X.shape[0]
        #print(X.shape)
        #print(Y.shape)
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()  # Reset gradients

        # AMP forward pass
        with autocast():
            outputs = model(X)  # Forward pass
            if unsqueezeY:
                loss = criterion(outputs, Y.unsqueeze(1))
            else:
                loss = criterion(outputs, Y)

        # Scale loss and backpropagation
        scaler.scale(loss).backward()

        # Optimizer step with scaled gradients
        scaler.step(optimizer)
        scaler.update()

        #outputs = model(X)
        #loss = criterion(outputs, Y)  # Compute loss
        #loss.backward()  # Backpropagation
        #optimizer.step()  # Update weights

        running_loss += loss.item()

    return running_loss / len(train_loader)  # Average loss

def evaluation_epoch(model, val_loader, criterion, device, desc, unsqueezeY = False):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # No gradients needed
        for X, Y in tqdm(val_loader, desc=desc, leave=False):
            #batch_size = X.shape[0]
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            if unsqueezeY:
                loss = criterion(outputs, Y.unsqueeze(1))
            else:
                loss = criterion(outputs, Y)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def multilabel_evaluate(model, loader, criterion, device, desc = 'testing'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # shape [batch_size, 2]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Apply sigmoid + threshold
            preds = (torch.sigmoid(outputs) > 0.5).float()  # shape [batch_size, 2]
            #preds = outputs

            # Count fully correct samples (both bits match)
            fully_correct = (preds == labels).all(dim=1).sum().item()
            correct += fully_correct

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct / total_samples

    # Concatenate predictions/labels for further analysis
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return epoch_loss, epoch_acc, all_preds, all_labels

def aggressive_arcsinh(x, center_region_size=80, upper_percentile=90):
    # x is assumed to be a tensor of shape [C, H, W]
    C, H, W = x.shape

    # Compute per-channel median
    medians = torch.median(x.reshape(C, -1), dim=1)[0]  # shape [C]

    # Replace values below the median with the median (broadcasted over H, W)
    x_new = torch.maximum(x, medians.reshape(C, 1, 1))
    
    # Define center region coordinates
    center_start_h = (H - center_region_size) // 2
    center_start_w = (W - center_region_size) // 2
    center_region = x_new[:, center_start_h:center_start_h+center_region_size, 
                            center_start_w:center_start_w+center_region_size]
    
    # Compute per-channel upper threshold from the center region
    upper_thresh = torch.quantile(center_region.reshape(C, -1), upper_percentile / 100.0, dim=1)  # shape [C]
    
    # Clamp values above the upper threshold
    x_clamped = torch.minimum(x_new, upper_thresh.reshape(C, 1, 1))
    
    # Apply arcsinh scaling
    return torch.asinh(x_clamped)

class SubsetWithTransform(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.new_transform = transform

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        if self.new_transform:
            sample = self.new_transform(sample)
        return sample, target