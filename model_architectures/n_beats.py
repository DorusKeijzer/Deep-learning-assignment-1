
import torch
from torch import nn
from model_architectures.base_model import BaseModel

class NBEATSModel(BaseModel):
    def __init__(self, input_size: int, stack_types: list = ["generic", "seasonal"], num_blocks: int = 3):
        super().__init__(input_size)
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.stacks = torch.nn.ModuleList([self._create_stack(stack_type) for stack_type in stack_types])
        
    def _create_stack(self, stack_type):
        blocks = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            blocks.append(NBEATSBlock(self.input_size, 64, 32, stack_type))
        return blocks
    
    @property
    def model_parameters(self):
        return f"stacks={'-'.join(self.stack_types)}, blocks={self.num_blocks}"
    
    @property
    def name(self):
        return f"NBEATS_{self.model_parameters}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        forecast = torch.zeros_like(x[:, -1:])
        
        for stack in self.stacks:
            for block in stack:
                backcast, fcast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + fcast
                
        return forecast

class NBEATSBlock(torch.nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, theta_dim: int, mode: str):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.theta = torch.nn.Linear(hidden_dim, theta_dim)
        self.backcast_mlp = torch.nn.Linear(theta_dim, input_size)
        self.forecast_mlp = torch.nn.Linear(theta_dim, input_size)
        
    def forward(self, x):
        h = self.fc(x)
        theta = self.theta(h)
        backcast = self.backcast_mlp(theta)
        forecast = self.forecast_mlp(theta)
        return backcast, forecast
