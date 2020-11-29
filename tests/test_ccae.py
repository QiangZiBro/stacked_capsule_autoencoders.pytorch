import torch
from model.models.ccae import CCAE, ConstellationEncoder, ConstellationDecoder


def test_ccae_forward():
    B = 4
    k = 11  # number of input set
    dim_input = 2  # 2 for 2D point
    dim_speical_features = 16
    n_votes = 4
    n_objects = 3

    model = CCAE(dim_input, n_objects, dim_speical_features, n_votes)
    print(model)
    x = torch.rand(B, k, dim_input)

    decoded_dict = model(x)
    op_matrix = decoded_dict.op_matrix
    ov_matrix = decoded_dict.ov_matrix
    standard_deviation = decoded_dict.standard_deviation
    part_presence = decoded_dict.part_presence
    object_presence = decoded_dict.object_presence

    assert ov_matrix.shape == (B, n_objects, dim_input, dim_input)
    assert op_matrix.shape == (B, n_objects, n_votes, dim_input)
    assert standard_deviation.shape == (B, n_objects, n_votes, 1)
    assert part_presence.shape == (B, n_objects, n_votes, 1)
    assert object_presence.shape == (B, n_objects, 1)

    return decoded_dict, x


def test_encoder_forward():
    B = 100
    k = 11
    dim_input = 2
    dim_speical_features = 16
    n_objects = 3

    x = torch.rand(B, k, dim_input)
    model = ConstellationEncoder(dim_input, n_objects, dim_speical_features)
    print(model)
    encoded_dict = model(x)
    ov_matrix = encoded_dict.ov_matrix
    special_features = encoded_dict.special_features
    object_presence = encoded_dict.object_presence

    assert ov_matrix.shape == (B, n_objects, dim_input, dim_input)
    assert special_features.shape == (B, n_objects, dim_speical_features)
    assert object_presence.shape == (B, n_objects, 1)


def test_decoder_forward():
    B = 100
    dim_input = 2
    dim_speical_features = 16
    n_objects = 3
    n_votes = 4

    x = torch.rand(B, n_objects, dim_speical_features)
    model = ConstellationDecoder(dim_input, n_objects, dim_speical_features, n_votes)
    decoded_dict = model(x)

    op_matrix = decoded_dict.op_matrix
    standard_deviation = decoded_dict.standard_deviation
    part_presence = decoded_dict.part_presence

    assert op_matrix.shape == (B, n_objects, n_votes, dim_input)
    assert standard_deviation.shape == (B, n_objects, n_votes, 1)
    assert part_presence.shape == (B, n_objects, n_votes, 1)
