from utils.dataloader import PaddingDataset
from model.models import models

for lag_parameter in range(1000): 
    for model in models:
        dataset = PaddingDataset(lag_parameter)
        for x, y in dataset:


