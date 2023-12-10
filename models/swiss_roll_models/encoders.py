import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import MLP



class EncoderVAE(nn.Module):
    """
    VAE encoder 
    """
    def __init__(self, dim_x,  gen_nodes_x, act_x, n_layers_x,
                dim_u, gen_nodes_u, act_u, n_layers_u, dim_z, 
                full_posterior=True, u_in_encoderx=False):
        super(EncoderVAE, self).__init__()

        # Encoder for ata x
        self.input_dim_x = dim_x
        self.gen_nodes_x = gen_nodes_x
        self.act_x = act_x
        self.n_layers_x = n_layers_x

        # Encoder for prior input u
        self.input_dim_u = dim_u
        self.gen_nodes_u = gen_nodes_u
        self.act_u = act_u
        self.n_layers_u = n_layers_u

        # Model architecture variance
        self.full_posterior = full_posterior
        self.u_in_encoderx = u_in_encoderx

        # Latent dimension
        self.latent_dim = dim_z

        # # Construct the encoders
        if self.u_in_encoderx:
            encoderx_input_dim = dim_x + dim_u
        else:
            encoderx_input_dim = dim_x
        self.encoder_x_mean = MLP(encoderx_input_dim, gen_nodes_x, dim_z, act_x, n_layers_x)
        self.encoder_x_logvar = MLP(encoderx_input_dim, gen_nodes_x, dim_z, act_x, n_layers_x)

    def forward(self, x, u):
        """
        Forward pass of the encoder
        """
        if self.u_in_encoderx:
            x = torch.cat([x, u], dim=-1)
        z_mean, z_logvar = self.encoder_x_mean(x), self.encoder_x_logvar(x)

        # The prior is from isotropic Gaussian
        lam_mean = torch.zeros_like(z_mean)
        lam_logvar = torch.zeros_like(z_logvar)

        return z_mean, z_logvar, lam_mean, lam_logvar
    

##############################################################################################################

class EncoderPiVAE(nn.Module):
    """
    IVAE encoder with exdriven mode and MLP encoders for both x and u.
    The default hyperparameters specify the original PiVAE.
    """
    def __init__(self, dim_x,  gen_nodes_x, act_x, n_layers_x,
                dim_u, gen_nodes_u, act_u, n_layers_u, dim_z, 
                full_posterior=True, u_in_encoderx=False):
        super(EncoderPiVAE, self).__init__()

        # Encoder for data x
        self.input_dim_x = dim_x
        self.gen_nodes_x = gen_nodes_x
        self.act_x = act_x
        self.n_layers_x = n_layers_x

        # Encoder for prior input u
        self.input_dim_u = dim_u
        self.gen_nodes_u = gen_nodes_u
        self.act_u = act_u
        self.n_layers_u = n_layers_u

        # Model architecture variance
        self.full_posterior = full_posterior
        self.u_in_encoderx = u_in_encoderx

        # Latent dimension
        self.latent_dim = dim_z

        # # Construct the encoders
        if self.u_in_encoderx:
            encoderx_input_dim = dim_x + dim_u
        else:
            encoderx_input_dim = dim_x
        self.encoder_x_mean = MLP(encoderx_input_dim, gen_nodes_x, dim_z, act_x, n_layers_x)
        self.encoder_x_logvar = MLP(encoderx_input_dim, gen_nodes_x, dim_z, act_x, n_layers_x)

        # # The prior encoder as two single-head MLP
        # self.encoder_u_mean = MLP(dim_u, gen_nodes_u, dim_z, act_u, n_layers_u)
        # self.encoder_u_logvar = MLP(dim_u, gen_nodes_u, dim_z, act_u, n_layers_u)

        # The prior encoder as a dual-head MLP 
        self.encoder_u = MLP(dim_u, gen_nodes_u, dim_z*2, act_u, n_layers_u)

    def forward(self, x, u):
        """
        Forward pass of the encoder
        """
        if self.u_in_encoderx:
            x = torch.cat([x, u], dim=-1)
        z_mean, z_logvar = self.encoder_x_mean(x), self.encoder_x_logvar(x)

        # lam_mean, lam_logvar = self.encoder_u_mean(u), self.encoder_u_logvar(u)

        # The prior encoder as a dual-head MLP
        lam_mean, lam_logvar = self.encoder_u(u).chunk(2, dim=-1)

        if self.full_posterior:
            z_mean, z_logvar = self.compute_posterior([z_mean, z_logvar, lam_mean, lam_logvar])

        return z_mean, z_logvar, lam_mean, lam_logvar
    
    def compute_posterior(self, args):
        """Compute the full posterior of q(z|x, u). We assume that q(z|x, u) \prop q(z|x)*p(z|u). Both q(z|x) and p(z|u) are Gaussian distributed.
        
        # Arguments
            args (list of tensors): mean and log of variance of q(z|x) and p(z|u)
            
        # Returns
            mean and log of variance of q(z|x, u) (list of tensors)
        """
        z_mean, z_log_var, lam_mean, lam_log_var = args

        # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2))
        post_mean = (z_mean / (1 + torch.exp(z_log_var - lam_log_var))) + (lam_mean / (1 + torch.exp(lam_log_var - z_log_var)))
        post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))

        return post_mean, post_log_var
