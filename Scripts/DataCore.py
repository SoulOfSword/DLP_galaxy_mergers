from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import torch

class RegressionDataset(Dataset):
    def __init__(self, datadir,labels,transform):
        self.lengt = len(labels.data)
        self.labels = labels
        
        self.transform = transform
        
        self.dataDir = datadir

    def __len__(self):
        return self.lengt

    def __getitem__(self, idx):
    
        label = self.labels.data[idx]
        
        tim = label['time_before_merger']
        ID = label["ID"]
        snap =label["snapnum"]

        img = fits.open(self.dataDir+f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits")[0].data
        img = img.astype(np.float32, copy=False)
        img = img.newbyteorder("=")  
        
        given = torch.tensor(self.transform(img))
        out = torch.tensor(tim)

        return given,out

    
class ClassificationDataset_labels(Dataset):
    def __init__(self, datadir, labels, transform):
        """
        Args:
          datadir (str): Base directory containing the FITS images.
          labels: FITS HDU (labels.data) with columns 'is_pre_merger', 'is_ongoing_merger', 'is_post_merger'.
          transform (callable): Function to apply to the raw image.
        """
        self.datadir = datadir
        self.labels = labels
        self.transform = transform
        self.length = len(self.labels.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        row = self.labels.data[idx]
        ID = row["ID"]
        snap = row["snapnum"]
        
        # Build file path for your FITS image
        file_path = os.path.join(
            self.datadir,
            f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/"
            f"JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits"
        )
        
        # Load FITS image
        with fits.open(file_path) as hdul:
            img_data = hdul[0].data.astype(np.float32).newbyteorder("=")
        
        # Apply the transformation
        img_transformed = self.transform(img_data)
        
        # Convert to tensor and replicate channel to get shape [3, H, W]
        img_tensor = torch.tensor(img_transformed).unsqueeze(0).repeat(3, 1, 1)
        
        # Read flags
        is_pre = (row['is_pre_merger'] == 1)
        is_ongoing = (row['is_ongoing_merger'] == 1)
        is_post = (row['is_post_merger'] == 1)
        
        # 3-class labeling logic:
        # 0 => non-merger
        # 1 => pre-merger
        # 2 => ongoing OR post
        if is_pre:
            class_label = 1
        elif is_ongoing or is_post:
            class_label = 2
        else:
            class_label = 0
        
        label_tensor = torch.tensor(class_label, dtype=torch.long)
        return img_tensor, label_tensor
    
    
class ClassificationArrayDataset(Dataset):
    def __init__(self, datadir, labels, transform):
        """
        Args:
          datadir (str): Base directory containing the FITS images.
          labels: FITS HDU (labels.data) with columns 'is_pre_merger', 'is_ongoing_merger', 'is_post_merger'.
          transform (callable): Function to apply to the raw image.
        """
        self.datadir = datadir
        self.labels = labels
        self.transform = transform
        self.length = len(self.labels.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        row = self.labels.data[idx]
        ID = row["ID"]
        snap = row["snapnum"]
        
        # Build file path for your FITS image
        file_path = os.path.join(
            self.datadir,
            f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/"
            f"JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits"
        )
        
        # Load FITS image
        with fits.open(file_path) as hdul:
            img_data = hdul[0].data.astype(np.float32).newbyteorder("=")
        
        # Apply the transformation
        img_transformed = self.transform(img_data)
        
        # Convert to tensor and replicate channel to get shape [3, H, W]
        img_tensor = torch.tensor(img_transformed).unsqueeze(0).repeat(3, 1, 1)
        
        # Read flags
        is_pre = (row['is_pre_merger'] == 1)
        is_ongoing = (row['is_ongoing_merger'] == 1)
        is_post = (row['is_post_merger'] == 1)
        
        # 3-class labeling logic, multiple things can be true:
        # 00 -> non merger
        # 10 -> pre merger
        # 01 -> post merger
        # 11 -> post merger, will merge again
        
        
        labelList = [0,0]
        
        
        if is_pre:
            labelList[0] = 1
        if is_ongoing or is_post:
            labelList[1] = 1
        
        
        label_tensor = torch.tensor(labelList, dtype=torch.int) #if there's a dtype error, change this to long
        return img_tensor, label_tensor