'''
This is a unifi_d training script intened to be used for all
IVAE models and on rat hippocampus dataset. 
'''
import os
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import matplotlib.pyplot as plt

from models.model_utils import make_and_restore_model
from models.loaders import DataLoaderWithCollate
from models import datasets, train, defaults, loaders
from utils import control_random_seed


import wandb


parser = ArgumentParser()
# TODO 0: Throw default values into defaults.py
parser.add_argument('-C', '--cache_dir', type=str, default='training_cache',
                    help="directory to store model outputs")
parser.add_argument('-L', '--lr', type=float, default=5e-4, help="learning rate")
parser.add_argument('-MO', '--mode', type=str, default='exdriven',
                    help="driven mode, (exdriven, indriven, or mixdriven)")
parser.add_argument('-M', '--model', type=str, default='pivae',
                    help="model identifier, see models/model_utils.py for supported")
parser.add_argument('-E', '--epochs', type=int, default=300,
                    help="number of epochs to train")
parser.add_argument('-S', '--seed', type=int, default=111, help="random seed")
parser.add_argument('-B', '--batch_size', type=int, default=1, help="batch size")

# Dataset config
parser.add_argument('-UL', '--u_label', type=str, default='loc_dir',
                    help="label for u, (loc_dir, loc_time, dir_time, time, location, direction, loc_dir_time)")

# Model architecture kwargs
parser.add_argument('-EX', '--encoder_x_arch', type=str, default='mlp')
parser.add_argument('-EU', '--encoder_u_arch', type=str, default='mlp')
parser.add_argument('-D', '--decoder_arch', type=str, default='gin')
parser.add_argument('-T', '--tau', type=float, default=1e-1,
                    help="softmax++ temperature hyperparameter")
parser.add_argument('-K', '--category_k', type=int, default=6,
                    help="number of categories for softmax++")

# Encoder architecture kwargs
parser.add_argument('-DX', '--dim_x', type=int, default=120)
parser.add_argument('-GNX', '--gen_nodes_x', type=int, default=60)
parser.add_argument('-AX', '--act_x', type=str, default='tanh')
parser.add_argument('-NX', '--n_layers_x', type=int, default=3)
parser.add_argument('-GNU', '--gen_nodes_u', type=int, default=20)
parser.add_argument('-AU', '--act_u', type=str, default='tanh')
parser.add_argument('-NU', '--n_layers_u', type=int, default=3)
parser.add_argument('-DZ', '--dim_z', type=int, default=2)
parser.add_argument('-FP', '--full_posterior', action='store_true',
                    help="indicator of whether to use full posterior")
parser.add_argument('-UIE', '--u_in_encoderx', action='store_true',
                    help="indicator of whether to use u in encoder_x")

# Decoder architecture kwargs
parser.add_argument('-NB', '--n_blk', type=int, default=2)
parser.add_argument('-MDL', '--mdl', type=str, default='poisson')
parser.add_argument('-MGN', '--min_gen_nodes', type=int, default=30)
parser.add_argument('-DD', '--dd', type=str, default=None)
parser.add_argument('-OS', '--output_scale', type=float,
                    default=1.0, help='output scalar multiplier')

# Meta config
parser.add_argument('-TR', '--test_run', action='store_true',
                    help="indicator of whether it is a test run")
parser.add_argument('-W', '--use_wandb', action='store_false',
                    help="indicator of whether not to use wandb")
parser.add_argument('-EID', '--exp_id', type=str, default='rat_hippo_test_runs',
                    help='experiment id for wandb')


def main(args, cache_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO 1: implement the dataset
    train_val_data = datasets.RatHippo(
        '../data/achilles_data/Achilles_data.mat', None, args.mode, args.u_label)  # all as train
    train_data = datasets.RatHippo(
        '../data/achilles_data/Achilles_data.mat', 'train', args.mode, args.u_label)
    val_data = datasets.RatHippo('../data/achilles_data/Achilles_data.mat',
                                 'val', args.mode, args.u_label)

    if args.decoder_arch.lower() == 'igr':
        train_data.add_trial_id_and_time_stamp_to_data()
        val_data.add_trial_id_and_time_stamp_to_data()
        train_val_data.add_trial_id_and_time_stamp_to_data()

        # val_data = train_data

    encoder_arch_kwargs = {
        'dim_x': args.dim_x,
        'gen_nodes_x': args.gen_nodes_x,
        'act_x': args.act_x,
        'n_layers_x': args.n_layers_x,
        'dim_u': train_data.dim_u,
        'gen_nodes_u': args.gen_nodes_u,
        'act_u': args.act_u,
        'n_layers_u': args.n_layers_u,
        'dim_z': args.dim_z,
        'full_posterior': args.full_posterior,
        'u_in_encoderx': args.u_in_encoderx
    }

    decoder_arch_kwargs = {
        'n_blk': args.n_blk,
        'dim_x': args.dim_x,
        'dim_z': args.dim_z,
        'mdl': args.mdl,
        'min_gen_nodes': args.min_gen_nodes,
        'dd': args.dd,
        'category_k': args.category_k,
        'tau': args.tau,
        'output_scale': args.output_scale,
        'trial_lengths': train_val_data.trial_lengths
    }

    arch_kwargs = {
        'model_arch': args.model,
        'encoder_x_arch': args.encoder_x_arch,
        'encoder_u_arch': args.encoder_u_arch,
        'decoder_arch': args.decoder_arch,
        'category_k': args.category_k,
        'tau': args.tau,
        'encoder_arch_kwargs': encoder_arch_kwargs,
        'decoder_arch_kwargs': decoder_arch_kwargs,
        'obs_dist': args.mdl
    }

    model = make_and_restore_model(arch=args.model.lower(), arch_kwargs=arch_kwargs)

    # BATCH_SIZE = args.batch_size
    BATCH_SIZE = 1  # now we have to use batch size 1 because of how we retrieve free parameter y in the decoder for igr

    train_loader = DataLoaderWithCollate(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoaderWithCollate(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_val_loader = DataLoaderWithCollate(
        train_val_data, batch_size=BATCH_SIZE, shuffle=False)

    train_kwargs = {
        'epochs': args.epochs,
        'lr': args.lr,
        'cache_dir': cache_dir,
        'device': device,
        'use_wandb': args.use_wandb,
    }

    train.train_model(model, train_loader, val_loader, **train_kwargs)

    # Latent visualization code. Refactor this into a separate analysis script later.
    model.eval()
    model.load_state_dict(torch.load(cache_dir + "/best_model.pt"))
    # Create an ordereddict to store the latent variables
    output_latent = OrderedDict([('z_mean', []), ('z_logvar', []),
                                ('lam_mean', []), ('lam_logvar', [])])
    # output_recon = OrderedDict(
    #     [('x', []), ('g(y)', []), ('g(y_mu)', [])])

    with torch.no_grad():
        for data in train_val_loader:
            if len(data) == 2:
                x, u = data
            elif len(data) == 4:
                x, u, trial_id, time_stamps = data

            z_mean, z_logvar, lam_mean, lam_logvar = model.encoder(
                x.to(device), u.to(device))

            # if model.decoder_arch.lower() == 'igr':
            #     z = model.reparameterize(z_mean, z_logvar)
            #     x_pred_y, x_pred_y_mu = model.decoder.forward_get_x_compare(
            #         z, trial_id[0], time_stamps)
            #     output_recon['x'].append(
            #         F.one_hot(x.long(), num_classes=args.category_k).float().cpu().numpy())
            #     output_recon['g(y)'].append(x_pred_y.cpu().numpy())
            #     output_recon['g(y_mu)'].append(x_pred_y_mu.cpu().numpy())

            output_latent['z_mean'].append(z_mean.cpu().numpy())
            output_latent['z_logvar'].append(z_logvar.cpu().numpy())
            output_latent['lam_mean'].append(lam_mean.cpu().numpy())
            output_latent['lam_logvar'].append(lam_logvar.cpu().numpy())

    output_latent = {k: np.concatenate(v) for k, v in output_latent.items()}
    # output_recon = {k: np.concatenate(v) for k, v in output_recon.items()}

    ############################
    ######## make plots ########
    ############################
    u_all = train_val_data.get_u_all(u_label='loc_dir')

    for key, latent in output_latent.items():
        plot_latent2d_rat_hippo(u_all, latent, title=key)



def get_tc_rd(y, hd, hd_bins):  # compute empirical tunning curve of data
    tuning_curve = np.zeros((len(hd_bins) - 1, y.shape[1]))
    for ii in range(len(hd_bins) - 1):
        data_pos = (hd >= hd_bins[ii]) * (hd <= hd_bins[ii + 1])
        tuning_curve[ii, :] = y[data_pos, :].mean(axis=0)
    return tuning_curve


def plot_latent2d_rat_hippo(u_all, latent_2d, title='', show_plot=True):
    """
    Plot the learned 2D latent of rat hippocampus dataset, with color representing direction and alpha representing location

    Args:
        u_all: list of np.array, len(u_all) = total number of trials. Each array has shape (num_samples, 3)
        latent_2d: np.array, shape (total number of samples, 2)
    """

    ll = 11
    hd_bins = np.linspace(0, 1.6, ll)
    select = np.concatenate(u_all)[:, 1] == 1
    tc1 = get_tc_rd(latent_2d[select], np.concatenate(u_all)[select, 0], hd_bins)
    select = np.concatenate(u_all)[:, 2] == 1
    tc2 = get_tc_rd(latent_2d[select], np.concatenate(u_all)[select, 0], hd_bins)

    dis_mat = np.zeros((len(tc1), len(tc2)))
    for jj in range(len(tc1)):
        dis_mat[jj] = np.sqrt(np.square(tc1[jj] - tc2).sum(axis=-1))

    ll = 5000
    # fig = plt.figure(figsize=(5.5, 4))
    # ax = plt.subplot(111)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fsz = 14

    # learn locations
    select = np.concatenate(u_all)[:ll, 1] == 1

    im = ax.scatter(
        latent_2d[:ll][select][:, 0],
        latent_2d[:ll][select][:, 1],
        s=1,
        c=np.concatenate(u_all)[:ll][select, 0],
        cmap="Reds",
        vmin=0,
        vmax=1.6,
    )
    ax.plot(tc1[:, 0], tc1[:, 1], c="black")
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # learn locations
    select = np.concatenate(u_all)[:ll][:, 1] == 0

    im = ax.scatter(
        latent_2d[:ll][select][:, 0],
        latent_2d[:ll][select][:, 1],
        s=1,
        c=np.concatenate(u_all)[:ll][select, 0],
        cmap="Blues",
        vmin=0,
        vmax=1.6,
    )

    ax.plot(tc2[:, 0], tc2[:, 1], c="black")
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax.set_xlabel("Latent 1", fontsize=fsz)
    ax.set_ylabel("Latent 2", fontsize=fsz)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=fsz)
    plt.setp(ax.get_yticklabels(), fontsize=fsz)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4, prune=None))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4, prune=None))

    plt.tight_layout()
    if title != '':
        plt.title(title)

    if args.use_wandb:
        wandb.log({title: wandb.Image(plt)})

    if show_plot:
        plt.show()


if __name__ == "__main__":
    args = parser.parse_args()

    run_id = f"{args.model}_encx{args.encoder_x_arch}_encu{args.encoder_u_arch}_dec{args.decoder_arch}_u{args.u_label}_fp{args.full_posterior}_uinx{args.u_in_encoderx}_{args.u_label}_tau{args.tau}_os{args.output_scale}_seed{args.seed}_lr{args.lr}_epochs{args.epochs}"
    print(run_id)
    control_random_seed(args.seed)

    # Initialize wandb
    if args.use_wandb:
        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project=args.exp_id,
            # Track hyperparameters and run metadata
            config=dict(args._get_kwargs()),
            notes=run_id,
            name=f"seed{args.seed}"
        )

    # Create output directory
    today = datetime.now().strftime("%m%d")
    cache_dir = f"{args.cache_dir}/{today}/{run_id}"

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    main(args, cache_dir)
