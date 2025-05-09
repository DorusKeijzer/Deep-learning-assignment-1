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
    def __init__(self, raw_data: np.ndarray, lag_size: int, scaler: StandardScaler, train_mean: float):
        # Pad with the training mean (not zeros) to avoid extreme scaled values
        self.train_mean = train_mean
        padded_raw = np.concatenate([np.full(lag_size, self.train_mean), raw_data])
        self.data = scaler.transform(padded_raw.reshape(-1, 1)).flatten().astype(np.float32)
        self.lag_size = lag_size
        
    def __len__(self):
        return len(self.data) - self.lag_size
    
    def __getitem__(self, idx: int):
        feature = self.data[idx : idx + self.lag_size]
        target = self.data[idx + self.lag_size]
        return feature, target

def create_datasets_and_loaders(
    lag_size: int, 
    use_padding: bool,
    test_ratio: float = 0.2, 
    batch_size: int = 32, 
    seed: int = 42
):
    # Load and split data
    raw_data = loadmat("data/Xtrain.mat")["Xtrain"].flatten().astype(np.float32)
    split_idx = int(len(raw_data) * (1 - test_ratio))
    raw_train, raw_test = raw_data[:split_idx], raw_data[split_idx:]

    # Compute training mean for padding
    train_mean = raw_train.mean()

    # Scale using train stats
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(raw_train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(raw_test.reshape(-1, 1)).flatten()

    # Create datasets
    if use_padding:
        train_ds = PaddingDataset(raw_train, lag_size, scaler, train_mean)
        test_ds = PaddingDataset(raw_test, lag_size, scaler, train_mean)
    else:
        train_ds = NonPaddingDataset(scaled_train, lag_size)
        test_ds = NonPaddingDataset(scaled_test, lag_size)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler
if __name__ == "__main__":
    # Create datasets with aligned splits
    padding_train_loader, padding_test_loader, _ = create_datasets_and_loaders(

         lag_size=5, 
        use_padding=True,
         test_ratio=0.2,
         batch_size=32
    )

    print(f"NonPadding Dataset - Train: {len(padding_train_loader.dataset)} samples")
    print(f"NonPadding Dataset - Test: {len(padding_test_loader.dataset)} samples")
    nonpadding_train_loader, nonpadding_test_loader, _ = create_datasets_and_loaders(

         lag_size=5, 
        use_padding=False,
         test_ratio=0.2,
         batch_size=32
    )

    print(f"Nonnonpadding Dataset - Train: {len(nonpadding_train_loader.dataset)} samples")
    print(f"Nonnonpadding Dataset - Test: {len(nonpadding_test_loader.dataset)} samples")

