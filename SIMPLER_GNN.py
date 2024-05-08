###########################
# This code was partially taken from GAMD (https://arxiv.org/abs/2112.03383, Li, Zijie and Meidani, Kazem and Yadav, Prakarsh and Barati Farimani, Amir, 2022.)
###########################

import numpy as np
import torch
import dgl.nn
import dgl.function as fn
from GAMD.md_module import get_neighbor
from sklearn.preprocessing import StandardScaler
import os, sys

from typing import List, Set, Dict, Tuple, Optional

import argparse
import os, sys
import joblib
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint, odeint_adjoint

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch.nn as nn

from nn_module import GAMD, GAMD_for_posvel

BOX_SIZE_n = 27.27
BOX_SIZE = np.array([BOX_SIZE_n, BOX_SIZE_n, BOX_SIZE_n])

pos_var = 61.97
pos_mean = 13.635

def cubic_kernel(r, re):
    eps = 1e-3
    r = torch.threshold(r, eps, re)
    return nn.ReLU()((1. - (r/re)**2)**3)


class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='softplus',
                 init_param=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'softplus':
            act_fn = nn.Softplus()
        elif activation == 'shifted_softplus':
            act_fn = nn.ShiftedSoftplus()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh, elu, gelu, silu, softplus, shifted_softplus as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if hidden_layer == 1:
                    if activation_first:
                        mlp_layer += [act_fn, nn.Linear(in_feats, out_feats)]
                    else:
                        print('Using MLP with no hidden layer and activations! Fall back to nn.Linear()')
                        mlp_layer += [nn.Linear(in_feats, out_feats)]
                elif not activation_first:
                    mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:   # l == hidden_layer-1
                mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)
        if init_param:
            self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat):
        return self.mlp_layer(feat)

class Full_predictor(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 conv_layer=4, 
                 edge_emb_dim=128,
                 use_layer_norm=True,
                 use_batch_norm=False, ##changed it
                 drop_edge=False,
                 augmentation = False,
                 pretrained = False,
                 ):
        super(Full_predictor, self).__init__()

        GNN = GAMD(
                  encoding_size = 128,
                  out_feats = 3,
                  hidden_dim = hidden_dim,
                  edge_embedding_dim = edge_emb_dim,
                  conv_layer = conv_layer,
                  drop_edge = drop_edge,
                  use_layer_norm = use_layer_norm,
                  box_size = BOX_SIZE,)
        if pretrained == True:
            pretrained_weights_path = './model_ckpt/GAMDwithJAX_new/checkpoint_30.ckpt'

            pretrained_state_dict = torch.load(pretrained_weights_path)['state_dict']

            adjusted_state_dict = {}

            for key, value in pretrained_state_dict.items():
                # Remove 'pnet_model.' prefix and ensure the keys match the model's expected keys
                new_key = key.replace("pnet_model.", "")
                adjusted_state_dict[new_key] = value

            GNN.load_state_dict(adjusted_state_dict, strict=False)

            # Freeze the weights of the pretrained part (force_predictor)
            for param in GNN.parameters():
                param.requires_grad = False
        
        self.integrationscheme = SONODE(GNN, augmentation)

    def forward(self, pos_vel, t):
        out = self.integrationscheme(pos_vel, t)
        return out

class Full_predictor_just_posvel(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 conv_layer=4,  # at some point try with 3
                 edge_emb_dim=128,
                 use_layer_norm=True,
                 use_batch_norm=False, ##changed it
                 drop_edge=False,
                 augmentation = False,
                 ):
        super(Full_predictor_just_posvel, self).__init__()

        GNN = GAMD_for_posvel(
                  encoding_size = 128,
                  out_feats = 3,
                  hidden_dim = hidden_dim,
                  edge_embedding_dim = edge_emb_dim,
                  conv_layer = conv_layer,
                  drop_edge = drop_edge,
                  use_layer_norm = use_layer_norm,
                  box_size = BOX_SIZE,
                  augmentation = augmentation)
                  
        self.justGNN = just_do_it(GNN)

    def forward(self, pos_vel, t):
        out = self.justGNN(pos_vel, t)
        return out


class just_do_it(nn.Module):
    def __init__(self, GNN):
        super(just_do_it, self).__init__()
        self.GAMD_for_posvel = GNN
    
    def forward(self, pos_vel, t):
        new_pos_vel = self.GAMD_for_posvel(denormalize_pos(pos_vel[:,:3]),pos_vel)
        return new_pos_vel

class SONODE(nn.Module):
    def __init__(self, force_predictor, augmentation):
        super(SONODE, self).__init__()
        if augmentation == True:
            self.fc = MLP(12, 9, hidden_dim = 128, hidden_layer=3, activation="softplus") ##aug
        if augmentation == False:
            self.fc = MLP(9, 6, hidden_dim = 128, hidden_layer=3, activation="softplus")
        self.force_predictor = force_predictor
    
    def forward(self, pos_vel, t):
        force = self.force_predictor(denormalize_pos(pos_vel[:,:3]))
        new_pos_vel = self.fc(torch.cat((pos_vel, force), dim=1))
        return new_pos_vel

def denormalize_pos(normalized):
    var_tensor = torch.tensor(pos_var, dtype=normalized.dtype, device=normalized.device).cuda()
    mean_tensor = torch.tensor(pos_mean, dtype=normalized.dtype, device=normalized.device).cuda()
    
    # Calculate standard deviation from variance
    std_dev_tensor = torch.sqrt(var_tensor)
    
    # Perform denormalization
    denormalized = normalized * std_dev_tensor + mean_tensor
    return denormalized


class GDEFunc(nn.Module):
    def __init__(self, gnn:nn.Module):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0
    
    def set_graph(self, g:dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.gnn(x, t)
        return x

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        

    def forward(self, integration_time, x):
        out = odeint_adjoint(self.odefunc, x, integration_time, atol=1e-6, rtol=1e-6, method ='rk4')
        #out = odeint(self.odefunc, x, integration_time, atol=1e-3, rtol=1e-3, method ='rk4')

        return out

class EntireModel(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                architecture,
                hidden_dim=128,
                conv_layer=4,
                edge_embedding_dim=128,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False,
                augmentation = False,
                pretrained = False,
                ):
        super(EntireModel, self).__init__()

        self.encoding_size = encoding_size
        self.architecture = architecture
        self.augmentation = augmentation

        if self.augmentation == True:
            self.finalboss = MLP(9,6, hidden_dim= 128, hidden_layer=2)

        if architecture == 'assisted':
            self.GNODE = ODEBlock(GDEFunc(Full_predictor(hidden_dim=hidden_dim,
                                                conv_layer=conv_layer,
                                                edge_emb_dim=edge_embedding_dim,
                                                use_layer_norm=use_layer_norm,
                                                use_batch_norm=not use_layer_norm,
                                                drop_edge=drop_edge,
                                                augmentation=augmentation,
                                                pretrained = pretrained)))
        
        if architecture == 'simple':
            self.GNODE = ODEBlock(GDEFunc(Full_predictor_just_posvel(hidden_dim=hidden_dim,
                                                conv_layer=conv_layer,
                                                edge_emb_dim=edge_embedding_dim,
                                                use_layer_norm=use_layer_norm,
                                                use_batch_norm=not use_layer_norm,
                                                drop_edge=drop_edge,
                                                augmentation=augmentation,)))
        


    def forward(self,
                fluid_pos_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                t,
                ):

        if self.augmentation == True:
            augmentation = torch.zeros_like(fluid_pos_lst[0])
            inp = torch.cat((fluid_pos_lst[0], fluid_vel_lst[0], augmentation), dim=1)

        else: inp = torch.cat((fluid_pos_lst[0], fluid_vel_lst[0]), dim=1)

        result = self.GNODE(t, inp)

        if self.augmentation == True:
            result = self.finalboss(result)

        pos_result = result[1:,:, :3]
        vel_result = result[1:,:, 3:]

        return pos_result, vel_result

# code from DGL documents
class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))
