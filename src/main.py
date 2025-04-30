from dataloader import PaddingDataset


for lag_parameter in range(1000): 
    dataset = PaddingDataset(lag_parameter)
