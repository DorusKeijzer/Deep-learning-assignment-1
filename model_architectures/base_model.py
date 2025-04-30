import torch
from abc import ABC, abstractmethod

class BaseModel(torch.nn.Module, ABC):
    """Abstract base class requiring a 'name' property and forward().""" 
    def __init__(self):
        super().__init__() 
        self.is_baseline = False


    @property
    @abstractmethod
    def name(self) -> str:
        """Mandatory property: Model name (must be implemented by child classes)."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mandatory forward pass."""
        pass

