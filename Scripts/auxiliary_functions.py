import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Aggressive arcsinh scaling
def aggressive_arcsinh_scaling(image):
    """
    Apply aggressive arcsinh scaling to enhance low surface brightness features.
    Steps:
      - Compute the arcsinh of the image.
      - Replace pixel values below the median with the median.
      - Clip high pixel values using the 90th percentile of the central region.
      - Normalize the image to the [0,1] range.
    """
    # If image has more than 2 dimensions; first dimension is the channel
    #print(f"{image.ndim}\n")
    if image.ndim > 2:
        image = image[0]
    
    image_scaled = np.arcsinh(image)
    median_val = np.median(image_scaled)
    image_scaled[image_scaled < median_val] = median_val

    # Define a central region (80x80 pixels)
    h, w = image_scaled.shape
    cx, cy = h // 2, w // 2
    central_region = image_scaled[max(cx - 40, 0):min(cx + 40, h), max(cy - 40, 0):min(cy + 40, w)]
    threshold = np.percentile(central_region, 90)

    image_scaled[image_scaled > threshold] = threshold

    # Normalize to [0, 1]
    image_norm = (image_scaled - np.min(image_scaled)) / (np.max(image_scaled) - np.min(image_scaled) + 1e-8)
    return image_norm

#TODO:
#Test binning effects on learning

def binningTransform(image: torch.Tensor, sizeBin: int = 1) -> torch.Tensor:  
    """
    Apply binning (averaging) to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        sizeBin (int, optional): Binning factor. Default is 1 (no binning).
    
    Returns:
        torch.Tensor: Binned image tensor of shape (C, H//sizeBin, W//sizeBin).
    """
    if sizeBin <= 1:
        return image  # No binning needed
    
    C, H, W = image.shape
    if H % sizeBin != 0 or W % sizeBin != 0:
        raise ValueError("Image dimensions must be divisible by binning factor.")
    
    # Reshape and average over binning regions
    image = image.view(C, H // sizeBin, sizeBin, W // sizeBin, sizeBin)
    binnedImage = image.mean(dim=(2, 4))
    
    return binnedImage

# Focal loss BCE 
class FocalLossBCE(nn.Module):
    """
    Multi-label Focal Loss built on BCEWithLogits:
      - gamma > 1 to down-weight well-classified examples
      - pos_weight can be used for further imbalance (like BCEWithLogitsLoss)
    """
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean', pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight  # can be a tensor of shape [2] if you have 2 bits

    def forward(self, inputs, targets):
        """
        inputs: shape [batch_size, 2], raw logits
        targets: shape [batch_size, 2], in {0,1}
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )  # shape [batch_size, 2]

        # p = sigmoid(logits)
        p = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)  # shape [batch_size, 2]

        # focal factor
        focal_factor = (1 - pt) ** self.gamma

        loss = focal_factor * bce_loss
        # alpha can be a scalar to scale the entire loss
        loss = self.alpha * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Dataset for Classification
class ClassificationDataset_values(Dataset):
    def __init__(self, datadir, labels, transform):
        """
        Args:
          datadir (str): Base directory containing the FITS images.
          labels: FITS HDU with a 'data' attribute.
          transform (callable): Function to apply to the raw image.
        """
        self.datadir = datadir
        self.labels = labels  # FITS HDU with structured data (labels.data)
        self.transform = transform
        self.length = len(self.labels.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        label_entry = self.labels.data[idx]
        ID = label_entry["ID"]
        snap = label_entry["snapnum"]
        # Here, we use the 'time_before_merger' column.
        # In the classification, we define three classes:
        # 0: non-merger, 1: pre-merger, 2: post-merger.
        tim = label_entry['time_before_merger']

        # file path
        file_path = os.path.join(
            self.datadir,
            f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits"
        )

        with fits.open(file_path) as hdul:
            img_data = hdul[0].data  # assuming primary HDU contains the image
        
        # Convert to float32 and ensure correct byte order
        img_data = img_data.astype(np.float32, copy=False)
        img_data = img_data.newbyteorder("=")
        
        # Apply transformation
        img_transformed = self.transform(img_data)
        
        # Convert to tensor and add a channel dimension (assumes grayscale image)
        img_tensor = torch.tensor(img_transformed).unsqueeze(0)  # shape: [1, H, W]

        # Replicate the single channel to create a 3-channel image: shape becomes [3, H, W]
        img_tensor = img_tensor.repeat(3, 1, 1)

        # Define the class label based on merger time.
        # For example, we adopt the paper’s default:
        # pre-merger: t_merger between -0.8 and -0.1 Gyr -> label 1
        # post-merger: t_merger between 0.1 and 0.3 Gyr -> label 2
        # Otherwise, non-merger -> label 0
        if (-0.8 <= tim <= -0.1):
            class_label = 1  # pre-merger
        elif (0.1 <= tim <= 0.3):
            class_label = 2  # post-merger
        else:
            class_label = 0  # non-merger
        
        label_tensor = torch.tensor(class_label, dtype=torch.long)
        return img_tensor, label_tensor

# Training+evaluation functions

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def multilabel_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)             # shape [batch_size, 2]
        loss = criterion(outputs, labels)   # BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Threshold each of the 2 outputs
        preds = (torch.sigmoid(outputs) > 0.5).float()  # shape [batch_size, 2]
        
        # If you want to count a sample correct only if *both bits* match:
        #   all_dims_correct = (preds == labels).all(dim=1)
        #   correct += all_dims_correct.sum().item()
        
        # Alternatively, you can count correctness bit by bit:
        bitwise_correct = (preds == labels).sum(dim=1)  # how many bits correct for each sample
        # e.g., if both bits match, bitwise_correct == 2. If one bit matches, bitwise_correct == 1.
        # For "fully correct" (both bits):
        fully_correct = (bitwise_correct == 2).sum().item()
        correct += fully_correct
    
    avg_loss = running_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc

def multilabel_evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # shape [batch_size, 2]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Apply sigmoid + threshold
            preds = (torch.sigmoid(outputs) > 0.5).float()  # shape [batch_size, 2]

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

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # Compute per-class precision and recall (do not average so you can see the performance for non-mergers, pre- and post-mergers)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return cm, acc, prec, rec

def evaluate_with_probabilities(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            out = model(inputs)
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def multilabel_evaluate_with_probabilities(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # For BCEWithLogitsLoss, each logit is for a separate bit (e.g. pre, post).
            out = model(inputs)             # shape [batch_size, 2]
            probs = torch.sigmoid(out)      # Convert logits -> probabilities in [0,1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def save_unique_figure(fig, directory=".", base_name="my_figure", ext="png"):
	os.makedirs(directory, exist_ok=True)
	i = 0
	while True:
		if i == 0:
			filename = os.path.join(directory, f"{base_name}.{ext}")
		else:
			filename = os.path.join(directory, f"{base_name}_{i}.{ext}")
		if not os.path.exists(filename):
			fig.savefig(filename)
			break
		i += 1

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, save=False, cmap=plt.cm.Blues):
    """
    From scikit-learn: plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.

	Taken from Duarte's plotting.py function
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"

    # Compute confusion matrix
    if len(y_true.shape) > 1 and len(y_pred.shape) > 1:
        cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    else:
        cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(title)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        ylabel="True label",
        xlabel="Predicted label",
    )

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = 0.5#cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
	#if save: save_unique_figure(fig, directory="../Plots", base_name="confusion_matrix")

def multilabel_plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plots a confusion matrix. If y_true and y_pred are multi-label
    (shape [N,2]), they will be mapped to combos 0..3 for a single 4x4 matrix.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    if not title:
        title = "Normalized confusion matrix" if normalize else "Confusion matrix"

    # Convert multi-label [N,2] -> single combo in [0..3]
    def combo_map(arr_2d):
        # arr_2d shape: [N, 2]
        # each row: [bit0, bit1] => bit0*2 + bit1
        return arr_2d[:, 0] * 2 + arr_2d[:, 1]

    if y_true.ndim == 2 and y_true.shape[1] == 2:
        y_true_1d = combo_map(y_true.astype(int))
    else:
        y_true_1d = y_true

    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        y_pred_1d = combo_map(y_pred.astype(int))
    else:
        y_pred_1d = y_pred

    cm = confusion_matrix(y_true_1d, y_pred_1d)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(title)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           ylabel="True label", xlabel="Predicted label")

    # Default combos if not provided
    if classes is None:
        classes = ["00", "01", "10", "11"]  # for 2-bit combos
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    fmt = ".2f" if normalize else "d"
    thresh = 0.5  # or cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()