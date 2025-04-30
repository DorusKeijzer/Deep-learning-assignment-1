import torch
from base_model import BaseModel

class MostRecentBaseline(BaseModel):
    """
    Baseline model that returns the most recent entry in the feature.
    """
    def name(self):
        return "Most Recent Baseline"

    def forward(self, x):
        return x[:,-1]

class MeanBaseline(BaseModel):
    """
    Baseline model that returns the mean of the feature.
    """
    def name(self):
        return "Mean Baseline"
    
    def forward(self, x):
        return torch.mean(x, dim=1)
