import torch as ch
import os
from . import rat_hippo_models, swiss_roll_models


def make_and_restore_model(*_, arch, arch_kwargs={}, dataset='rat_hippo'):
    """
    """

    # TODO: implement checkpoint loading

    if dataset == 'rat_hippo':
        return rat_hippo_models.__dict__[arch](**arch_kwargs)
    elif dataset == 'swiss_roll':
        return swiss_roll_models.__dict__[arch](**arch_kwargs)

