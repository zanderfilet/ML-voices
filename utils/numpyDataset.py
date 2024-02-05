import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    
    def __init__(self, file_X, file_Y):
        
        self.Y = np.load(file_Y)
        self.X = np.load(file_X) 

    def __getitem__(self, index):
        
        Y = self.Y[index]
        X = self.X[index]
        
        if X.max() > 400:
            X = X / 1024
        
        return X, Y
        

    def __len__(self):
        
        return len(self.Y)