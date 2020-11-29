import torch.nn as nn
from base import BaseModel
from model.modules.setmodules import ISAB, SAB, PMA


class SetTransformer(BaseModel):
    """"""

    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=True,
    ):
        """Set Transformer, An autoencoder model dealing with set data

        Input set X with N elements, each `dim_input` dimensions, output
        `num_outputs` elements, each `dim_output` dimensions.

        In short, choose:
        N --> num_outputs
        dim_input --> dim_output

        Hyper-parameters:
            num_inds
            dim_hidden
            num_heads
            ln
        Args:
            dim_input: Number of dimensions of one elem in input set X
            num_outputs: Number of output elements
            dim_output: Number of dimensions of one elem in output set
            num_inds: inducing points number
            dim_hidden: output dimension of one elem of middle layer
            num_heads: heads number of multi-heads attention in MAB
            ln: whether to use layer norm in MAB
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        """
        Args:
            X: (B, N, dim_input)

        Returns:
            output set with shape (B, num_outputs, dim_output)
        """
        return self.dec(self.enc(X))
