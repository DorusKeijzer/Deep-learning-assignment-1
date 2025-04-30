import torch
from model_architectures.base_model import BaseModel

class MostRecentBaseline(BaseModel):
    """
    baseline model that returns the most recent entry in the feature.
    """
    def __init__(self, input_size):
        super().__init__(input_size)
        self.is_baseline = True # true for baseline models only

    @property
    def name(self):
        return "most recent baseline"
    
    def forward(self, x):
        return x[:,-1]

class MeanBaseline(BaseModel):
    """
    Baseline model that returns the mean of the feature.
    """
    def __init__(self, input_size):
        super().__init__(input_size)
        self.is_baseline = True # true for baseline models only

    @property
    def name(self):
        return "Mean Baseline"
    
    def forward(self, x):
        return torch.mean(x, dim=1)
