
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .encoders import EncoderPiVAE, EncoderVAE
from .decoders import DecoderGIN


def compute_posterior(args):
    """Compute the posterior distribution q(z|x, u) = N(z|post_mean, post_log_var).
    Used in the original Pi-VAE implementation"""
    z_mean, z_log_var, lam_mean, lam_log_var = args

    # Compute posterior mean and log variance
    post_mean = (z_mean / (1 + torch.exp(z_log_var - lam_log_var))) + \
        (lam_mean / (1 + torch.exp(lam_log_var - z_log_var)))
    post_log_var = z_log_var + lam_log_var - \
        torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))

    return post_mean, post_log_var


# loss functions
def reconstruction_loss(x, firing_rate, distribution='gaussian'):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            firing_rate, x, reduction='sum').div(batch_size)
    elif distribution == 'weighted_bernoulli':
        weight = torch.tensor([0.1, 0.9]).to("cuda")  # just a label here
        weight_ = torch.ones(x.shape).to("cuda")
        weight_[x <= 0.5] = weight[0]
        weight_[x > 0.5] = weight[1]
        recon_loss = F.binary_cross_entropy_with_logits(firing_rate, x, reduction='none')
        recon_loss = torch.sum(weight_ * recon_loss).div(batch_size)
    elif distribution == 'gaussian':
        firing_rate = F.sigmoid(firing_rate)
        recon_loss = F.mse_loss(firing_rate, x, reduction='sum').div(batch_size)
    elif distribution == 'poisson':
        firing_rate.clamp(min=1e-7, max=1e7)
        recon_loss = torch.sum(firing_rate - x * torch.log(firing_rate)).div(batch_size)
    elif distribution == 'poisson2':
        firing_rate = firing_rate + 1e-7
        recon_loss = torch.sum(firing_rate - x * torch.log(firing_rate)).div(batch_size)
    elif distribution == 'categorical':
        x = F.one_hot(x.long(), num_classes=firing_rate.shape[-1])
        recon_loss = -torch.sum(x * torch.log(firing_rate)).div(batch_size)
    else:
        raise NotImplementedError

    return recon_loss


def kl_divergence(z_mean, z_log_var, lam_mean, lam_log_var):
    batch_size = z_mean.size(0)
    assert batch_size != 0

    kl_loss = 1 + z_log_var - lam_log_var - \
        ((torch.square(z_mean - lam_mean) + torch.exp(z_log_var)) / torch.exp(lam_log_var))

    kl_loss = torch.sum(kl_loss, dim=-1)
    kl_loss = kl_loss * -0.5
    kl_loss = torch.mean(kl_loss)

    return kl_loss


class IVAE(nn.Module):
    """Define an IVAE model.
    """

    def __init__(self, model_arch, encoder_x_arch, encoder_u_arch, decoder_arch,
                 encoder_arch_kwargs, decoder_arch_kwargs, tau=None, category_k=None,
                 latent_dynamic=False, obs_dist=None, **kwargs):
        super(IVAE, self).__init__()
        # Prepare the parameters
        self.model_arch = model_arch
        self.encoder_x_arch = encoder_x_arch
        self.encoder_u_arch = encoder_u_arch
        self.decoder_arch = decoder_arch
        self.latent_dynamic = latent_dynamic
        self.obs_dist = obs_dist
        if tau is not None:
            self.tau = torch.tensor(tau)
        if category_k is not None:
            self.k = torch.tensor(category_k)

        self.encoder_arch_kwargs = encoder_arch_kwargs
        self.decoder_arch_kwargs = decoder_arch_kwargs

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def get_encoder(self):
        """
        Return the IVAE encoder specified by `encoder_x_arch`, `encoder_u_arch`, and `latent_dynamic`.
        """
        if self.latent_dynamic:
            raise NotImplementedError('latent_dynamic=True is not implemented yet.')
        else:  # feedforward
            if self.encoder_x_arch.lower() == 'mlp':
                if self.encoder_u_arch.lower() == 'mlp':
                    if self.model_arch.lower() in ['pivae', 'civae', 'igr_ivae']:
                        encoder = EncoderPiVAE(**self.encoder_arch_kwargs)
                    elif self.model_arch.lower() == 'vae':
                        encoder = EncoderVAE(**self.encoder_arch_kwargs)
                    else:
                        raise NotImplementedError(
                            'model_arch={} is not implemented yet.'.format(self.model_arch))
                elif self.encoder_u_arch.lower() == 'embedding':
                    raise NotImplementedError('encoder_u=embedding is not implemented yet.')

        return encoder

    def get_decoder(self):
        """return a decoder which takes the latent variable z as input 
        """
        if self.decoder_arch.lower() == 'mlp':
            raise NotImplementedError('decoder_arch=mlp is not implemented yet.')
            # decoder = DecoderMLP(kwargs)
        elif self.decoder_arch.lower() == 'gin':
            decoder = DecoderGIN(**self.decoder_arch_kwargs)

        return decoder

    # def forward(self, data, u=None):
    def forward(self, data, u=None):
        # handle change in implementation, refactor it our later
        if len(data) == 4:
            data, u, trial_id, time_stamps = data
        elif len(data) == 2:
            data, u = data
        else:
            assert type(data) != tuple, "data is not a tuple"

        z_mean, z_logvar, lam_mean, lam_logvar = self.encoder(data, u)
        z = self.reparameterize(z_mean, z_logvar)

        if self.decoder_arch.lower() == 'gin':
            y = self.decoder(z)

        constraint_loss = None
        log_jacob_g = None
        recon_loss = reconstruction_loss(data, y, distribution=self.obs_dist)

        kl_loss = kl_divergence(z_mean, z_logvar, lam_mean, lam_logvar)

        if constraint_loss is not None:
            if log_jacob_g is not None:
                return y, recon_loss, kl_loss, constraint_loss, log_jacob_g
            else:
                return y, recon_loss, kl_loss, constraint_loss
        else:
            return y, recon_loss, kl_loss

    def reparameterize(self, z_mean, z_log_var):
        # Get batch size and dimension of the latent space
        batch = z_mean.size(0)
        dim = z_mean.size(1)

        # Create a random tensor with Gaussian distribution (mean=0, std=1)
        epsilon = torch.randn(batch, dim).to(device=z_mean.device)

        # Reparameterization trick: z = mean + std * epsilon
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z
