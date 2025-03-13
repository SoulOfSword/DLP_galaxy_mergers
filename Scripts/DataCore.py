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

    
class ClassificationDataset(Dataset):
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
        
        isMerging = tim>0

        img = fits.open(self.dataDir+f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits")[0].data
        img = img.astype(np.float32, copy=False)
        img = img.newbyteorder("=")  
        
        given = torch.tensor(self.transform(img))
        out = torch.tensor(isMerging)

        return given,out