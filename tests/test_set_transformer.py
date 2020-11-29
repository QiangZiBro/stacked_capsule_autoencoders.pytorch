import torch
from model.models.set_transformer import SetTransformer


def test_forward():
    B = 1
    M = 24
    dummy_x = torch.rand(B, M, 144)

    dim_input = 144
    num_out_capsules = 20
    set_out = 100  # dim for each part capsule
    set_head = 12
    st = SetTransformer(
        dim_input, num_out_capsules, set_out, num_heads=set_head, dim_hidden=16, ln=True
    )

    assert st(dummy_x).shape == (B, num_out_capsules, set_out)
