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

class NonPaddingDataset(Dataset):
    """
    Does not pad the dataset with zeroes, i.e. it start at the `lag_size`th entry (fewer entries in total)
    
    i.e. with a lag size of 5, the first entry will be `x = [y_0,y_1,y_2,y_3,y_4], y = y_5`
    """
    def __init__(self, lag_size: int):
        self.data = np.array(loadmat("data/Xtrain.mat")["Xtrain"]).flatten()
        self.lag_size = lag_size
        
    def __len__(self):
        return len(self.data) - self.lag_size  # Only valid sequences
    
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError
        feature = self.data[idx : idx+self.lag_size]
        target = self.data[idx+self.lag_size]
        return feature.astype(np.float32), target.astype(np.float32)


class PaddingDataset(NonPaddingDataset): 
    """
    Pads the start of the dataset with a number of zeroes equal to the lag parameter. 

    i.e. with a lag size of 5, the first entry will be `x = [0,0,0,0,0], y = y_0`

    """ 
    def __init__(self, lag_size: int):
        raw_data = np.array(loadmat("data/Xtrain.mat")["Xtrain"]).flatten()
        self.data = np.concatenate([np.zeros(lag_size), raw_data])
        self.lag_size = lag_size
        
    def __len__(self):
        return len(self.data) - self.lag_size  # Same calculation but includes padding

def create_datasets_and_loaders(lag_size: int, test_ratio: float = 0.2, batch_size: int = 32, seed: int = 42):
    """
    Creates aligned train/test splits for both Padding and NonPadding datasets.
    
    Args:
        lag_size: Size of the lookback window
        test_ratio: Fraction of data to use for testing (0.0-1.0)
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (padding_train_loader, padding_test_loader, 
                 nonpadding_train_loader, nonpadding_test_loader)
    """
    raw_data = np.array(loadmat("data/Xtrain.mat")["Xtrain"]).flatten()
    n_total = len(raw_data)
    
    # Create datasets
    padding_ds = PaddingDataset(lag_size)
    nonpadding_ds = NonPaddingDataset(lag_size)
    
    # Calculate split sizes
    test_size = int(n_total * test_ratio)
    train_size = n_total - test_size
    
    # Padding dataset splits (same as original data)
    padding_train_size = train_size
    padding_test_size = test_size
    
    # Non-padding dataset splits
    nonpadding_total = len(nonpadding_ds)  # This equals n_total - lag_size
    nonpadding_test_size = test_size
    nonpadding_train_size = nonpadding_total - nonpadding_test_size

    # Create splits
    padding_train, padding_test = random_split(
        padding_ds,
        [padding_train_size, padding_test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    nonpadding_train, nonpadding_test = random_split(
        nonpadding_ds,
        [nonpadding_train_size, nonpadding_test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create DataLoaders
    loaders = (
        DataLoader(padding_train, batch_size=batch_size, shuffle=True),
        DataLoader(padding_test, batch_size=batch_size, shuffle=False),
        DataLoader(nonpadding_train, batch_size=batch_size, shuffle=True),
        DataLoader(nonpadding_test, batch_size=batch_size, shuffle=False)
    )
    
    return loaders


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


