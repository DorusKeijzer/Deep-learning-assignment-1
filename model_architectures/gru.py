import torch
from torch import nn
from model_architectures.base_model import BaseModel


class GRUModel(BaseModel):
    """
    parameters: 
    - hidden_dim: dimension of the hidden layers
    - num_layers: number of layers
    """
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int = 1):
        super().__init__(input_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)

    @property
    def name(self) -> str:
        return f"GRU, hidden dimensions: {self.hidden_dim}, number of layers: {self.num_layers}" 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has shape [batch_size, seq_len, input_size]
        if len(x.shape) == 2:  # [batch_size, features]
            x = x.unsqueeze(1)  # [batch_size, 1, features]
        
        batch_size = x.size(0)
        
        # Initialize hidden state with proper shape
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the last time step's output
        out = self.fc(out[:, -1, :])
        return out.squeeze()  # Remove extra dimension to match target shape
