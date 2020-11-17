import torch
from model.modules.neural import BatchLinear,BatchMLP


def test_batch_linear_forward():
    bl = BatchLinear(2, 3, 10)
    print(bl)
    x = torch.rand(4, 10, 2)
    assert bl(x).shape == (4, 10, 3)


def test_batch_mlp_forward():
    bl = BatchMLP(2, 3, 10, [128])
    print(bl)
    x = torch.rand(4, 10, 2)
    assert bl(x).shape == (4, 10, 3)
