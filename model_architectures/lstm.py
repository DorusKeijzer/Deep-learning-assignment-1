import torch
from torch import nn
from model_architectures.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_dim, num_layers, output_size=1):
        super(LSTMModel, self).__init__(input_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    @property
    def name(self) -> str:
        return f"LSTM"
    @property
    def model_parameters(self) -> str:
        return f"hidden dimensions: {self.hidden_dim}, number of layers: {self.num_layers}"