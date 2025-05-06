import torch
from torch import nn
from model_architectures.base_model import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__(input_size)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder = torch.nn.Linear(input_size, d_model)
        self.decoder = torch.nn.Linear(d_model, 1)
        
    @property
    def model_parameters(self):
        return f"d_model={self.d_model}, nhead={self.nhead}, layers={self.num_layers}"
    
    @property
    def name(self):
        return f"Transformer_{self.model_parameters}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return x
