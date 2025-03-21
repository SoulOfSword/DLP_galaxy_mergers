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

    
class BinaryClassificationDataset(Dataset):
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

        with fits.open(self.dataDir+f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits") as hdul:
            img = hdul[0].data
        img = img.astype(np.float32, copy=False)
        img = img.newbyteorder("=")  
        
        given = self.transform(torch.tensor(img))
        out = torch.tensor(np.float32(isMerging))

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
        
        ID = label["ID"]
        snap =label["snapnum"]

        #non = label['is_major_merger'] == 0
        pre = label['is_pre_merger'] == 1
        post = label['is_post_merger'] == 1 or label['is_ongoing_merger'] == 1
        isMerging = [pre, post] #, non

        with fits.open(self.dataDir+f"mock_v4/F150W/L75n1820TNG/snapnum_0{snap}/xy/JWST_50kpc_F150W_TNG100_sn0{snap}_xy_broadband_{ID}.fits") as hdul:
            img = hdul[0].data
        img = img.astype(np.float32, copy=False)
        img = img.newbyteorder("=")  
        
        given = self.transform(torch.tensor(img))
        out = torch.tensor(isMerging, dtype=torch.float32)

        return given,out

