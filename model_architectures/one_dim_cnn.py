
import torch

from torch import nn
from model_architectures.base_model import BaseModel
class CNN1DModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        num_channels: list = [32, 64],
        kernel_sizes: list = [3, 3],
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        pooling: str = 'max',
        pooling_sizes: list = [2, 2],
        adaptive_pooling: bool = True
    ):
        super().__init__(input_size)
        
        # Store parameters
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.pooling = pooling
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling_sizes = pooling_sizes
        self.adaptive_pooling = adaptive_pooling
        self.stride = stride
        self.dilation = dilation

        # Validate parameters
        assert len(num_channels) == len(kernel_sizes) == len(pooling_sizes)

        # Calculate output sizes upfront
        self.min_sequence_length = self._calculate_min_sequence_length()
        
        # Build layers
        layers = []
        in_channels = input_size
        
        for i in range(len(num_channels)):
            layers.append(self._build_conv_block(i, in_channels))
            in_channels = num_channels[i]

        self.cnn = nn.Sequential(*layers)
        self.fc = nn.LazyLinear(1)

    def _calculate_min_sequence_length(self):
        """Calculate minimum required input sequence length"""
        min_length = 1
        for ks, pool in zip(self.kernel_sizes, self.pooling_sizes):
            min_length = ((min_length + 2 * ((ks - 1) * self.dilation // 2) - self.dilation * (ks - 1)) // self.stride)
            min_length = max(min_length // pool, 1)
        return min_length

    def _build_conv_block(self, layer_idx, in_channels):
        """Build a complete convolution block with safety checks"""
        block = nn.Sequential()
        
        # Convolution
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.num_channels[layer_idx],
            kernel_size=self.kernel_sizes[layer_idx],
            stride=self.stride,
            dilation=self.dilation,
            padding=self._calculate_padding(self.kernel_sizes[layer_idx], self.dilation)
        )
        block.add_module(f"conv_{layer_idx}", conv)

        # Batch norm
        if self.use_batch_norm:
            block.add_module(f"bn_{layer_idx}", nn.BatchNorm1d(self.num_channels[layer_idx]))

        # Activation
        block.add_module(f"relu_{layer_idx}", nn.ReLU())

        # Dropout
        if self.dropout > 0:
            block.add_module(f"dropout_{layer_idx}", nn.Dropout(self.dropout))

        # Pooling with safety
        pool_size = self.pooling_sizes[layer_idx]
        if pool_size > 1:
            if self.adaptive_pooling:
                # Use adaptive pooling if requested
                block.add_module(f"adapool_{layer_idx}", nn.AdaptiveAvgPool1d(1))
            else:
                # Regular pooling with size validation
                pool_class = nn.MaxPool1d if self.pooling == 'max' else nn.AvgPool1d
                block.add_module(f"pool_{layer_idx}", pool_class(pool_size))

        return block

    def _calculate_padding(self, kernel_size, dilation):
        return (kernel_size - 1) * dilation // 2

    @property
    def name(self) -> str:
        return (f"1D CNN, channels: {self.num_channels}, "
                f"kernels: {self.kernel_sizes}, "
                f"pooling: {self.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shaping
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [batch, features, 1]
        elif x.shape[1] != self.input_size:
            x = x.permute(0, 2, 1)

        # Forward pass
        x = self.cnn(x)
        
        # Global average pooling
        x = x.mean(dim=-1)
        
        return self.fc(x).squeeze()

