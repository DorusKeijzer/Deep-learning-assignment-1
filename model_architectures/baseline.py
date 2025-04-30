import torch

class MostRecentBaseline(torch.nn.Module):
    """
    Baseline model that returns the most recent entry in the feature.
    """
    def forward(self, x):
        return x[-1]

class MeanBaseline(torch.nn.Module):
    """
    Baseline model that returns the mean of the feature.
    """
    def forward(self, x):
        return torch.mean(x)
