import torch
from torch import nn
from base import BaseModel
from monty.collections import AttrDict

class BatchLinear(BaseModel):
    """Performs k independent linear transformations of k vectors."""
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.linears = nn.ModuleList(nn.Linear(in_channels, out_channels)
                                for _ in range(k))

    def forward(self, x):
        """
        Args:
            x: (B, k, in_channels)

        Returns:
            (B, k, out_channels)
        """
        assert self.k == x.shape[1]
        result = []
        for i, linear in enumerate(self.linears):
            result.append(linear(x[:,i,:]))
        result = torch.stack(result, dim=1)
        return result

class BatchMLP(BaseModel):
    """
    Applies k independent MLPs on k inputs
    """
    def __init__(self, in_channels, out_channels, k, hidden):
        super().__init__()
        self.bmlp = nn.Sequential(*[
            BatchLinear(_in, _out, k)
            for _in, _out in zip([in_channels] + hidden, hidden + [out_channels])
        ])

    def forward(self, x):
        """

        Args:
            x: (B, k, in_channels)

        Returns:
            (B, k, out_channels)
        """
        return self.bmlp(x)