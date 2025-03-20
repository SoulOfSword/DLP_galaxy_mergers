import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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