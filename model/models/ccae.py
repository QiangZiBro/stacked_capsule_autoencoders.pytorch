import torch
import torch.nn.functional as F

from .set_transformer import SetTransformer
from ..modules.neural import BatchMLP
from torch import nn
from base import BaseModel
from monty.collections import AttrDict

class CCAE(BaseModel):
    """ Constellation capsule auto-encoder

    Input a point set, encode the set with Set Transformer to
    object capsules, and then decode to parts.

    Each object capsule has such components:
        OV matrix: (dim_input,dim_input)
        special features: (dim_speical_features,)
        presence: scalar

    Decoded part capsule has:
        OP matrix: (dim_input, 1)
        standard deviation: scalar
        presence: scalar

    """
    def __init__(self,
                 dim_input,
                 n_objects=3,
                 dim_speical_features=16,
                 n_votes=4,
                 **kargs):
        """

        Args:
            dim_input:int, number of points of input set
            n_objects:int, number of object capsules.
            dim_speical_features: #dimension of speical features of each object.
            n_votes: int, number of votes(or predicted parts) for each object.
            **kargs:
        """
        super().__init__()

        self.encoder = ConstellationEncoder(dim_input, n_objects, dim_speical_features)
        self.decoder = ConstellationDecoder(dim_input, n_objects, dim_speical_features, n_votes)

    def forward(self, x):
        """
        Args:
            x: a set with (B, M, dim_input)

        Returns:
            part capsules
        """
        res = self.encoder(x)

        special_features = res.special_features
        decoded_dict = self.decoder(special_features)

        res.update(decoded_dict)
        # add some useful infomation to result dict
        res.dim_input = self.encoder.dim_input
        return res

class ConstellationEncoder(BaseModel):
    """
    Input a point set, encode the set with Set Transformer to
    object capsules
    """
    def __init__(self, dim_input, n_objects, dim_speical_features, **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self.n_objects = n_objects
        self.dim_speical_features = dim_speical_features

        self.set_transformer = SetTransformer(
            dim_input=dim_input,
            num_outputs=n_objects,
            dim_output=dim_input**2+dim_speical_features+1,
            **kwargs
        )

    def forward(self, x):
        """

        Args:
            x: a set with (B, M, dim_input)

        Returns:
            OV matrix: (B, n_objects, dim_input, dim_input)
            special features: (B, n_objects, dim_speical_features)
            presence: scalar: (B, n_objects, 1)
        """
        B = x.shape[0]
        objects = self.set_transformer(x)  # (B, n_objects, dim_input**2+dim_speical_features+1)
        splits = [self.dim_input**2,self.dim_input**2+self.dim_speical_features]

        ov_matrix, special_features, presence = objects[:,:,:splits[0]],\
                                                 objects[:,:,splits[0]:splits[1]],\
                                                 objects[:,:,splits[1]:]

        ov_matrix = ov_matrix.reshape(B, self.n_objects, self.dim_input, self.dim_input)
        presence = F.softmax(presence, dim=1)
        return AttrDict(ov_matrix=ov_matrix,
                        special_features=special_features,
                        object_presence=presence)


class ConstellationDecoder(BaseModel):
    """
    Use mlp to decode each object's special features to parts.
    mlp architecture:
        dim_speical_features --> hidden --> n_votes*(dim_input+1+1)

    Notes: Notice that original implementation use BatchMLP:
    https://github.com/google-research/google-research/blob/8c9034fb956e0fda9a450d9cdacef2e70a9c9564/stacked_capsule_autoencoders/capsules/neural.py#L98-L130
    So we also implemented a batch mlp here
    """
    def __init__(self, dim_input, n_objects, dim_speical_features, n_votes, **kwargs):
        """
        Args:
            dim_input:int, number of points of input set
            n_objects:int, number of object capsules.
            dim_speical_features: #dimension of speical features of each object.
            n_votes: int, number of votes(or predicted parts) for each object.
        """
        super().__init__()
        hidden = [128]
        self.dim_speical_features = dim_speical_features
        self.n_votes = n_votes
        self.dim_input = dim_input


        self.bmlp = BatchMLP(dim_speical_features, n_votes*(dim_input+1+1), n_objects, hidden)

    def forward(self, x):
        """

        Args:
            x: special features, (B, n_objects, dim_speical_features)

        Returns:
            OP matrix:(B, n_objects, n_votes, dim_input)
            standard_deviation:(B, n_objects, n_votes, 1)
            presence:(B, n_objects, n_votes, 1)
        """
        B, n_objects, dim_speical_features = x.shape
        assert self.dim_speical_features == dim_speical_features


        x = self.bmlp(x) # (B, n_objects, n_votes*(dim_input+1+1))
        x_chunk = x.chunk(self.n_votes, dim=-1)
        x_object_part = torch.stack(x_chunk, dim=2) # (B, n_objects, n_votes, (dim_input+1+1))

        splits = [self.dim_input, self.dim_input+1]
        op_matrix = x_object_part[:,:,:,:splits[0]]
        standard_deviation = x_object_part[:,:,:,splits[0]:splits[1]]
        presence = x_object_part[:,:,:,splits[1]:]
        presence = F.softmax(presence, dim=2)
        return AttrDict(
            op_matrix=op_matrix,
            standard_deviation=standard_deviation,
            part_presence=presence
        )