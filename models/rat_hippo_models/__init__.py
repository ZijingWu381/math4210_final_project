import torch.nn as nn

from .ivae import IVAE
from torch.nn import Module


def get_ivae_model(model_arch, encoder_x_arch, encoder_u_arch, decoder_arch, 
                   encoder_arch_kwargs=None, decoder_arch_kwargs=None, tau=None, category_k=None, obs_dist=None,
                   pretrained=True, map_location='cpu', weightsdir_path=None, 
                    **kwargs):
    """
    Returns a pretrained/initialized IVAE model.
    Select pretrained=True for returning a model with pretrained weights.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """
    if pretrained:
        raise NotImplementedError('Pretrained models are not yet supported')
    else:
        model = globals()[f'IVAE'](model_arch, encoder_x_arch, encoder_u_arch, decoder_arch, 
                                   encoder_arch_kwargs, decoder_arch_kwargs, tau, category_k, obs_dist=obs_dist, **kwargs)
    return model

########## Define Getter Functions for Each IVAE's ##########
def pi_vae(pretrained=False, **kwargs):
    """
    Constructs a PIVAE model.
    """

    model_arch = kwargs.get('model_arch')
    encoder_x_arch = kwargs.get('encoder_x_arch', 'mlp')
    encoder_u_arch = kwargs.get('encoder_u_arch', 'mlp')
    decoder_arch = kwargs.get('decoder_arch', 'gin')
    tau = None
    category_k = None
    obs_dist = kwargs.get('obs_dist', None)
    encoder_arch_kwargs = kwargs.get('encoder_arch_kwargs')
    decoder_arch_kwargs = kwargs.get('decoder_arch_kwargs')

    decoder_arch_kwargs['category_k'] = None
    decoder_arch_kwargs['tau'] = None
    
    print(kwargs)

    model = get_ivae_model(model_arch, encoder_x_arch, encoder_u_arch,
                           decoder_arch, encoder_arch_kwargs,
                           decoder_arch_kwargs, 
                           tau, category_k, obs_dist=obs_dist, pretrained=pretrained)
    return model 

def vae_getter(pretrained=False, **kwargs):
    """
    Constructs a VAE model.
    """

    model_arch = kwargs.get('model_arch')
    encoder_x_arch = kwargs.get('encoder_x_arch', 'mlp')
    encoder_u_arch = kwargs.get('encoder_u_arch', 'mlp')
    decoder_arch = kwargs.get('decoder_arch', 'gin')
    tau = None
    category_k = None
    obs_dist = kwargs.get('obs_dist', None)
    encoder_arch_kwargs = kwargs.get('encoder_arch_kwargs')
    decoder_arch_kwargs = kwargs.get('decoder_arch_kwargs')

    decoder_arch_kwargs['category_k'] = None
    decoder_arch_kwargs['tau'] = None
    
    print(kwargs)

    model = get_ivae_model(model_arch, encoder_x_arch, encoder_u_arch,
                           decoder_arch, encoder_arch_kwargs,
                           decoder_arch_kwargs, 
                           tau, category_k, obs_dist=obs_dist, pretrained=pretrained)

    return model 

vae = vae_getter
pivae = pi_vae
