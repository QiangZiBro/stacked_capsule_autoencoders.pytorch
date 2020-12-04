import torch
import torch.nn.functional as F
from utils.math_ops import GuassianMixture as gmm


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ccae_loss(res_dict, target, epsilon=1e-6):
    """

    Args:
        res_dict:
        target: input set with (B, k, dim_input)
        epsilon: avoiding nan for reciprocal of standard deviation
    Returns:
        log likelihood for input dataset(here "target") , (B,)
    """
    # retrieve the variable (Sorry for possible complication)
    op_matrix = res_dict.op_matrix  # (B, n_objects, n_votes, dim_input)
    ov_matrix = res_dict.ov_matrix  # (B, n_objects, dim_input, dim_input)
    standard_deviation = res_dict.standard_deviation  # (B, n_objects, n_votes, 1)
    object_presence = res_dict.object_presence  # (B, n_objects, 1)
    part_presence = res_dict.part_presence  # (B, n_objects, n_votes, 1)
    dim_input = res_dict.dim_input
    B, n_objects, n_votes, _ = standard_deviation.shape
    op_matrix = op_matrix[:, :, :, :, None]  # (B, n_objects, n_votes, dim_input,1)
    ov_matrix = ov_matrix[:, :, None, :, :]  # (B, n_objects, 1, dim_input,dim_input)

    standard_deviation = epsilon + standard_deviation[Ellipsis, None]
    mu = ov_matrix @ op_matrix  # (B, n_objects, n_votes, dim_input,1)
    identity = (
        torch.eye(dim_input)
        .repeat(B, n_objects, n_votes, 1, 1)
        .to(standard_deviation.device)
    )
    sigma = identity * (
        1 / standard_deviation
    )  # (B, n_objects, n_votes, dim_input,dim_input)

    # (B, k, n_objects, n_votes)
    likelihood = gmm(mu, sigma).likelihood(
        target, object_presence=object_presence, part_presence=part_presence
    )
    log_likelihood = torch.log(likelihood.sum((1, 2, 3))).mean()

    res_dict.likelihood = likelihood  # data distribution predicted
    res_dict.log_likelihood = -log_likelihood  # loss

    return res_dict
