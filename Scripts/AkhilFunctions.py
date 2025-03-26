import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score

scaler = GradScaler()

import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that increases linearly during warmup, then decays following a cosine function.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps to linearly increase the learning rate.
        num_training_steps: The total number of training steps.
        num_cycles: The number of cosine cycles during decay (default is 0.5).
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        A LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, math.cos(math.pi * num_cycles * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def training_epoch(model, train_loader, optimizer, criterion, device, scheduler = None, unsqueezeY = False):
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
        if scheduler != None:
            scheduler.step()
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

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Concatenate predictions/labels for further analysis
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return epoch_loss, epoch_acc, all_preds, all_labels, balanced_acc

def nonbinary_multilabel_evaluate(model, loader, criterion, device, desc='testing'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)  # Ensure labels are integer class indices
            outputs = model(inputs)  # shape: [batch_size, 3]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Apply softmax and then take argmax for the predicted class
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct += (preds == labels).sum().item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct / total_samples
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, balanced_acc

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

class SubsetWithTransform_v2(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.new_transform = transform

    def __getitem__(self, idx):
        # Use get_raw to bypass the original transforms.
        sample, target = self.dataset.get_raw(self.indices[idx])
        if self.new_transform:
            sample = self.new_transform(sample)
        # Convert target to tensor.
        target = torch.tensor(target, dtype=torch.long)
        return sample, target