from model.loss import ccae_loss
from .test_ccae import test_ccae_forward


def test_ccae_loss():
    res_dict, x = test_ccae_forward()
    loss = ccae_loss(res_dict, x).log_likelihood
    assert loss.shape == (), "please check batch size or loss size"
