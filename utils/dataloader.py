from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
import numpy as np
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NonPaddingDataset(Dataset):
    def __init__(self, data: np.ndarray, lag_size: int):
        self.data = data.astype(np.float32)
        self.lag_size = lag_size
        
    def __len__(self):
        return len(self.data) - self.lag_size
    
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError
        feature = self.data[idx : idx + self.lag_size]
        target = self.data[idx + self.lag_size]
        return feature, target

class PaddingDataset(Dataset):
    def __init__(self, data: np.ndarray, lag_size: int):
        self.data = np.concatenate([np.zeros(lag_size, dtype=np.float32), data.astype(np.float32)])
        self.lag_size = lag_size
        
    def __len__(self):
        return len(self.data) - self.lag_size
    
    def __getitem__(self, idx: int):
        feature = self.data[idx : idx + self.lag_size]
        target = self.data[idx + self.lag_size]
        return feature, target

def create_datasets_and_loaders(
    lag_size: int, 
    test_ratio: float = 0.2, 
    batch_size: int = 32, 
    seed: int = 42
):
    # Load and split raw data
    raw_data = loadmat("data/Xtrain.mat")["Xtrain"].flatten().astype(np.float32)
    n_total = len(raw_data)
    split_idx = int(n_total * (1 - test_ratio))
    
    # Time-based split (no shuffling)
    raw_train = raw_data[:split_idx]
    raw_test = raw_data[split_idx:]
    
    # Scale the entire series
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(raw_train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(raw_test.reshape(-1, 1)).flatten()
    
    # Create datasets with correct padding/scaling order
    padding_train_ds = PaddingDataset(scaled_train, lag_size)
    padding_test_ds = PaddingDataset(scaled_test, lag_size)
    nonpadding_train_ds = NonPaddingDataset(scaled_train, lag_size)
    nonpadding_test_ds = NonPaddingDataset(scaled_test, lag_size)
    
    # Create DataLoaders
    loaders = (
        DataLoader(padding_train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(padding_test_ds, batch_size=batch_size, shuffle=False),
        DataLoader(nonpadding_train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(nonpadding_test_ds, batch_size=batch_size, shuffle=False)
    )
    
    return loaders, scaler

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

    # Create datasets with aligned splits
    padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = create_datasets_and_loaders(
         lag_size=5, 
         test_ratio=0.2,
         batch_size=32
    )

    print(f"Padding Dataset - Train: {len(padding_train_loader.dataset)} samples")
    print(f"Padding Dataset - Test: {len(padding_test_loader.dataset)} samples")
    print(f"NonPadding Dataset - Train: {len(nonpadding_train_loader.dataset)} samples")
    print(f"NonPadding Dataset - Test: {len(nonpadding_test_loader.dataset)} samples")


