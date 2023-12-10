"""This file implements the functional utils for the rat-hippo model modules"""""
import torch


def apply_along_axis(function, x, dim: int = 0):
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=dim)
    ], dim=dim)

# def apply_along_axis(x, tau, dim: int = 0, delta=1.0):
#     t = [
#         softmax_pp(x_i, tau, delta) for x_i in torch.unbind(x, dim=dim)
#     ]
#     return torch.stack(t, dim=dim)
#     # return torch.stack([
#     #     function(x_i) for x_i in torch.unbind(x, dim=dim)
#     # ], dim=dim)


def softmax_pp(y, tau, delta=1.0):
    """
    Args:
        y (tensor): shape (n_samples, n_categories)
    """
    # Compute softmax_++ with logsumexp.
    # Here's reference for softmax: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    # We extended it for softmax_++ 
    # TODO add derivation math in comment

    if y.dim() == 1:
        y = y.unsqueeze(1)

    n_samples = y.shape[0]
    sigma = torch.log(torch.tensor(delta / n_samples))
    ones = sigma * torch.ones(n_samples, 1)
    y_over_tau = y / tau
    concated = torch.cat((y_over_tau, ones), dim=1)
    ln_z = y / tau - torch.logsumexp(concated, dim=1, keepdim=True)

    z = ln_z.exp()
    z = torch.cat((z, 1 - z.sum(dim=1, keepdim=True)), dim=1)
    # 1 - z.sum(dim=1, keepdim=True) might give small negative values 
    # ~ -1e7 due to numerical error
    z = torch.clamp(z, min=1e-7)

    return z