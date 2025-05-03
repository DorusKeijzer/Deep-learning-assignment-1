import torch
from torch import nn
import torch.nn.functional as F
from model_architectures.base_model import BaseModel
from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # First causal convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation  # Causal padding
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Downsample if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        # First convolution
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Trim padding from right
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Downsample residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = residual[:, :, :out.size(2)]  # Match lengths

        out += residual
        return out


class TCNModel(BaseModel):
    def __init__(self, input_size: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__(input_size)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []
        in_channels = input_size  # Use lag parameter as input channels

        for i, out_channels in enumerate(num_channels):
            layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,  # Exponential dilation
                    dropout=dropout
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.final_fc = nn.Linear(in_channels, 1)

    @property
    def name(self) -> str:
        return "TCN"

    @property
    def model_parameters(self) -> str:
        return f"channels: {self.num_channels}, kernel: {self.kernel_size}, dropout: {self.dropout}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, input_size]
        if x.dim() == 2:
            x = x.unsqueeze(2)  # Add channel dimension

        # Permute to [batch_size, input_size, seq_len]
        x = x.permute(0, 2, 1)

        # Pass through TCN
        x = self.tcn(x)

        # Take last time step and apply final FC
        x = x[:, :, -1]
        return self.final_fc(x).squeeze()