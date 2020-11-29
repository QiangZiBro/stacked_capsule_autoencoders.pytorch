import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from monty.collections import AttrDict


class BatchLinear(BaseModel):
    """Performs k independent linear transformations of k vectors."""

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.linears = nn.ModuleList(
            nn.Linear(in_channels, out_channels) for _ in range(k)
        )

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
            result.append(linear(x[:, i, :]))
        result = torch.stack(result, dim=1)
        return result


class BatchMLP(BaseModel):
    """
    Applies k independent MLPs on k inputs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        k,
        hidden,
        activation=nn.ReLU,
        activate_final=False,
    ):
        super().__init__()

        blinears = []
        LAST = len([in_channels] + hidden) - 1
        for i, (_in, _out) in enumerate(
            zip([in_channels] + hidden, hidden + [out_channels])
        ):
            blinears.append(BatchLinear(_in, _out, k))
            if i == LAST and activate_final:
                blinears.append(activation())
            else:
                blinears.append(activation())

        self.bmlp = nn.Sequential(*blinears)

    def forward(self, x):
        """

        Args:
            x: (B, k, in_channels)

        Returns:
            (B, k, out_channels)
        """
        return self.bmlp(x)
