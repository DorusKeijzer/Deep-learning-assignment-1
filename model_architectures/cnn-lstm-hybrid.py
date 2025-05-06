
import torch
from torch import nn
from model_architectures.base_model import BaseModel

class CNNLSTMModel(BaseModel):
    def __init__(self, input_size: int, cnn_channels: list = [32, 64], kernel_size: int = 3, 
                 lstm_hidden: int = 64):
        super().__init__(input_size)
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        
        self.conv_layers = torch.nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.conv_layers.append(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'))
            in_channels = out_channels
            
        self.lstm = torch.nn.LSTM(input_size * cnn_channels[-1], lstm_hidden, batch_first=True)
        self.fc = torch.nn.Linear(lstm_hidden, 1)
        
    @property
    def model_parameters(self):
        return f"cnn={self.cnn_channels}, kernel={self.kernel_size}, lstm={self.lstm_hidden}"
    
    @property
    def name(self):
        return f"CNN-LSTM_{self.model_parameters}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Add channel dimension
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        _, (h_n, _) = self.lstm(x.reshape(batch_size, x.size(1), -1))
        return self.fc(h_n[-1])
