import torch
from torch import nn
from model_architectures.base_model import BaseModel


class GRUModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)

    @property
    def name(self) -> str:
        return f"GRU, hidden_dimensions: {self.hidden_dim}, number of layers: {self.num_layers}" 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the last time step's output
        out = self.fc(out[:, -1, :])
        return out

