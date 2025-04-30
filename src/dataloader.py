from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

"""
Pads the start of the dataset with a number of zeroes equal to the lag parameter. 

i.e. with a lag size of 5, the first entry will be x = [0,0,0,0,0] y = y_0

"""
class PaddingDataset(Dataset):
    def __init__(self, lag_size: int):
        self.data = np.array(loadmat("data/Xtrain.mat")["Xtrain"])
        self.data = np.append(np.zeros(lag_size), self.data)
        self.lag_size = lag_size

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx: int):
        target = self.data[idx+1+self.lag_size]
        feature = self.data[idx: idx+self.lag_size]

        return feature, target

if __name__ == "__main__":
    ds = PaddingDataset(3)
    for i, d in ds:
        print(i, d)

"data/Xtrain.mat"
