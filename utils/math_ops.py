import torch
import math
import torch.nn.functional as F

class GuassianMixture(object):
    """
    GMM for part capsules
    """
    def __init__(self, mu, sigma):
        """
        Args:
            mu: (B, n_objects, n_votes, dim_input, 1)
            sigma: (B, n_objects, n_votes, dim_input, dim_input)

        After initialized:
            mu:   (B, 1, n_objects, n_votes, dim_input, 1)
            sigma:(B, 1, n_objects, n_votes, dim_input,dim_input)
            multiplier:(B, 1, n_objects, n_votes, 1, 1)
        """
        #  Converse shape to
        #  (Batch_size, num_of_points, num_of_objects, number_of_votes, ...)

        mu = mu[:, None, ...]  # (B, 1, n_objects, n_votes, dim_input, 1)
        sigma = sigma[:,None,...] # (B, 1, n_objects, n_votes, dim_input,dim_input)

        self.sigma = sigma
        self.mu = mu
        self.sigma_inv = sigma.inverse()
        D = self.sigma.shape[-1]
        sigma_det = torch.det(sigma) # (B, 1, n_objects, n_votes)
        self.multiplier = 1/( (2*math.pi)**(D/2) * sigma_det.sqrt())[...,None,None]



    def likelihood(self, x, object_presence=None, part_presence=None):
        """ Compute likelihood of input set(a.k.a. x) given gaussian
        distribution with mu and sigma, also, we have to consider
        object_presence and part_presence.

        shape check:
        mu:   (B, 1, n_objects, n_votes, dim_input,1)
        sigma and sigma_inv:
              (B, 1, n_objects, n_votes, dim_input,dim_input)
        x:    (B, k, 1,         1,       dim_input,1)
        x-mu: (B, k, n_objects, n_votes, dim_input,1)
        (x-mu).transpose(-1,-2):
              (B, k, n_objects, n_votes, 1,dim_input)
        (x-mu).transpose(-1,-2)@sigma_inv@(x-mu):
              (B, k, n_objects, n_votes, 1,1)
        multiplier:(B, 1, n_objects, n_votes, 1, 1)

        Args:
            x: (B, k, dim_input)
            object_presence: (B, n_objects, 1) if exists
            part_presence: (B, n_objects, n_votes, 1) if exists
        Returns:
            pdf for input point set x
            (B, k, n_objects, n_votes)
        """
        x = x[:,:,None,None,:,None] # (B,k,1,1,dim_input,1)
        diff = x-self.mu
        exp_result = torch.exp(-0.5 * diff.transpose(-1, -2) @ self.sigma_inv @ diff)
        if object_presence is  not None and part_presence is not None:
            # (B, 1, n_objects, 1, 1, 1)
            object_presence = object_presence[:,None,...,None,None]
            # (B, 1, n_objects, n_votes, 1, 1)
            part_presence = part_presence[:,None,...,None]

            object_presence = torch.abs(object_presence)+1e-6
            part_presence = torch.abs(part_presence)+1e-6
            # object_presence = F.relu(object_presence)+1e-6
            # part_presence = F.relu(part_presence)+1e-6
            # object_presence[object_presence<0]=1e-6
            # part_presence[part_presence<0]=1e-6
            denominator = object_presence.sum(dim=2,keepdim=True)*part_presence.sum(dim=3,keepdim=True)

            exp_result = (object_presence*part_presence/denominator)*exp_result

        gaussian_likelihood = self.multiplier * exp_result

        return gaussian_likelihood.squeeze(-1).squeeze(-1)


    def plot(self, choose):
        raise NotImplemented