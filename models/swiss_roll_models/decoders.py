"""
This file contains the decoder models for the swiss roll dataset model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import MLP
from .utils import apply_along_axis, softmax_pp
import torch.distributions as tds
import numpy as np

########### GIN decoder ###########

class FirstNFlowLayer(nn.Module):
    def __init__(self, dim_x, dim_z, min_gen_nodes=30):
        super(FirstNFlowLayer, self).__init__()
        self.gen_nodes = max(min_gen_nodes, dim_x // 4)
        self.dim_z = dim_z
        self.layers = MLP(dim_z, self.gen_nodes, dim_x - dim_z, act='relu', n_layers=3)

    def forward(self, z_input):
        # output = z_input
        # for layer in self.layers:
        #     output = layer(output)
        output = self.layers(z_input)
        output = torch.cat((z_input, output), dim=-1)
        return output


class AffineCouplingLayer(nn.Module):
    def __init__(self, DD, min_gen_nodes=30, dd=None):
        super(AffineCouplingLayer, self).__init__()
        self.DD = DD  # Dim of input, which is dim_x since we are using Normalizing Flow
        self.dd = dd if dd is not None else self.DD // 2  # Dim of input which is fed to the st_layers
        self.s_dim = self.DD - self.dd - 1  # Dim of s

        n_nodes = [max(min_gen_nodes, self.DD // 4), max(min_gen_nodes,
                                                         self.DD // 4), 2 * (self.DD - self.dd) - 1]

        n_layers = 3
        # act_func = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
        act_func = [nn.ReLU()] * (n_layers-1) + [nn.Identity()]
        self.st_layers = MLP(self.dd, n_nodes[0], n_nodes[2], act=act_func, n_layers=n_layers)

    def forward(self, layer_input):
        x_input1 = layer_input[:, :self.dd]
        x_input2 = layer_input[:, self.dd:]
        st_output = x_input1

        st_output = self.st_layers(st_output)
        s_output = st_output[:, :self.s_dim]
        t_output = st_output[:, self.s_dim:]
        s_output = self.clamp_func(s_output)  # make sure output of s is small
        s_output = torch.cat([s_output, self.sum_func(s_output)],
                             dim=-1)  # enforce the last layer has sum 0

        assert (torch.mean(s_output, dim=-1) < 1e-7).all(), torch.mean(s_output, dim=-1)

        # Perform transformation
        trans_x = x_input2 * torch.exp(s_output) + t_output
        output = torch.cat((trans_x, x_input1), dim=-1)

        return output

    def clamp_func(self, x):
        return 0.1 * torch.tanh(x)

    def sum_func(self, x):
        return torch.sum(-x, dim=-1, keepdim=True)


class AffineCouplingBlock(nn.Module):
    """Define affine_coupling_block, which contains two affine_coupling_layer. 
    """

    def __init__(self, DD, min_gen_nodes=30, dd=None):
        super(AffineCouplingBlock, self).__init__()
        self.affine_coupling_layer1 = AffineCouplingLayer(DD, min_gen_nodes, dd)
        self.affine_coupling_layer2 = AffineCouplingLayer(DD, min_gen_nodes, dd)

    def forward(self, x_output):
        """
        Returns
            output (tensor): output of a GIN block (affine_coupling_block).
        """
        x_output = self.affine_coupling_layer1(x_output)
        x_output = self.affine_coupling_layer2(x_output)
        return x_output


class DecoderGIN(nn.Module):
    """Define mean(p(x|z)) using GIN volume preserving flow. 
    """

    def __init__(self, n_blk, dim_x, dim_z, mdl, min_gen_nodes=30, dd=None, category_k=None, tau=None, output_scale=1.0, learn_var=False,
                 trial_lengths=None):
        super(DecoderGIN, self).__init__()

        # self.n_blk = n_blk
        self.n_blk = n_blk

        self.mdl = mdl
        self.min_gen_nodes = min_gen_nodes
        self.dd = dd
        self.permute_ind = None  # To be set during forward pass
        self.dim_z = dim_z
        if category_k is not None:
            self.k = category_k
            self.km1 = category_k - 1
            self.dim_x = dim_x * self.km1
        else:
            self.dim_x = dim_x

        self.output_scale = output_scale

        self.learn_var = learn_var

        self.first_nflow_layer = FirstNFlowLayer(self.dim_x, dim_z, min_gen_nodes)
        self.affine_coupling_blocks = nn.Sequential(
            *[AffineCouplingBlock(self.dim_x, min_gen_nodes, dd) for _ in range(n_blk)])

    def forward(self, z_input):
        # Generate permutation indices
        if self.permute_ind is None:
            self.permute_ind = []
            for ii in range(self.n_blk):
                # TODO Implemented in the original pivae code. Whether it is necessary to set seed here?
                np.random.seed(ii)
                # self.permute_ind.append(torch.randperm(self.dim_x, device=z_input.device))
                self.permute_ind.append(torch.from_numpy(
                    np.random.permutation(self.dim_x)).to(z_input.device))

        output = self.first_nflow_layer(z_input)

        # Apply permutation and affine_coupling_block
        for ii in range(self.n_blk):
            output = output[:, self.permute_ind[ii]]
            output = self.affine_coupling_blocks[ii](output)

        if self.mdl == 'poisson' or self.mdl == 'gumbel':
            output = F.softplus(output)
            output = torch.clamp(output, min=1e-7, max=1e7)
            # output = F.sigmoid(output)
        elif self.mdl == 'categorical':
            # output = F.softplus(output)
            # output = torch.clamp(output, min=1e-7, max=1e7)
            # output = softmax_pp(output, tau=1e-2, delta=1.0)
            tau = 1
            delta = 1
            batch_size = output.shape[0]
            output = output.view(batch_size, self.dim_x // self.km1, self.km1)
            output = apply_along_axis(lambda x: softmax_pp(x, tau, delta), output, dim=0)
        elif self.mdl == 'gaussian':
            if not self.learn_var:
                # minus 0.5 to make output contains negative values, it is an empirical intuition
                # output = F.sigmoid(output) - 0.5
                # output = torch.clamp(output, min=-1e1, max=1e1)
                # print(f"output min {output.min()} max {output.max()} mean {output.mean()}")
                # output = nn.LeakyReLU()(output)

                # minus 0.5 to make output contains negative values, it is an empirical intuition
                output = self.output_scale * (F.sigmoid(output) - 0.5)
            else:
                output = output 

        return output

