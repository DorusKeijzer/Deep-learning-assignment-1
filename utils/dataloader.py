from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

"""
Does not pad the dataset with zeroes, i.e. it start at the `lag_size`th entry (fewer entries in total)
"""
class NonPaddingDataset(Dataset):
    def __init__(self, lag_size: int):
        self.data = np.array(loadmat("data/Xtrain.mat")["Xtrain"])
        self.lag_size = lag_size

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx: int):
        # nth data point
        target = self.data[idx+self.lag_size]
        # (n - lagsize)th data point unttil (n-1)th data point
        feature = self.data[idx : idx+self.lag_size]

        return feature, target


"""
Pads the start of the dataset with a number of zeroes equal to the lag parameter. 

i.e. with a lag size of 5, the first entry will be x = [0,0,0,0,0] y = y_0

"""
class PaddingDataset(NonPaddingDataset):
    def __init__(self, lag_size: int):
        super().__init__(lag_size)
        # Pad dataset with zeroes
        self.data = np.append(np.zeros(lag_size), self.data)

if __name__ == "__main__":
    ds = PaddingDataset(3)
    print("padding:")
    for i, (x, y) in enumerate(ds):
        print(i, x, y)
        if i == 10:
            break
    ds = NonPaddingDataset(3)
    print("non padding:")
    for i, (x, y) in enumerate(ds):
        print(i, x, y)
        if i == 10:
            break



