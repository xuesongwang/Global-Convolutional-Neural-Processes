"""Module for neural networks"""
import abc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from npf.architectures import MLP, merge_flat_input
from npf.utils.helpers import (
    MultivariateNormalDiag,
    isin_range,
)
from npf.utils.initialization import weights_init

from .helpers import pool_and_replicate_middle

__all__ = [
    "NeuralNetwork"
]


class NeuralNetwork(abc.ABC):

    def __init__(
        self,
        ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.is_heteroskedastic = is_heteroskedastic

        if x_transf_dim is None:
            self.x_transf_dim = self.x_dim
        elif x_transf_dim == -1:
            self.x_transf_dim = self.r_dim
        else:
            self.x_transf_dim = x_transf_dim

        if XEncoder is None:
            XEncoder = self.dflt_Modules["XEncoder"]

        if Decoder is None:
            Decoder = self.dflt_Modules["Decoder"]

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)

        # times 2 out because loc and scale (mean and var for gaussian)
        self.decoder = Decoder(self.x_transf_dim, self.y_dim * 2)

        self.PredictiveDistribution = PredictiveDistribution
        self.p_y_loc_transformer = p_y_loc_transformer
        self.p_y_scale_transformer = p_y_scale_transformer

        self.reset_parameters()



    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given a set of context feature-values {(x^c, y^c)}_c and target features {x^t}_t, return
        a set of posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, x_dim]
            Set of all context features {x_i}. Values need to be in interval [-1,1]^d.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_dim]
            Set of all target features {x_t}. Values need to be in interval [-1,1]^d.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        # self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

        # size = [batch_size, *n_cntxt, x_transf_dim]
        # size = [batch_size, *n_trgt, x_transf_dim]
        X_trgt = self.x_encoder(X_trgt)
        X_trgt = X_trgt.unsqueeze(0) # n_samples
        p_yCc = self.decode(X_trgt)

        return p_yCc, None, None, None


    def decode(self, X_trgt):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        R_trgt : torch.Tensor, size=[n_z_samples, batch_size, *n_trgt, r_dim]
            Set of all target representations {r^t}_t.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        """
        # size = [n_z_samples, batch_size, *n_trgt, y_dim*2]
        p_y_suffstat = self.decoder(X_trgt)
        # size = [n_z_samples, batch_size, *n_trgt, y_dim]
        p_y_loc, p_y_scale = p_y_suffstat.split(self.y_dim, dim=-1)

        p_y_loc = self.p_y_loc_transformer(p_y_loc)
        p_y_scale = self.p_y_scale_transformer(p_y_scale)

        #! shuld probably pool before p_y_scale_transformer
        if not self.is_heteroskedastic:
            # to make sure not heteroskedastic you pool all the p_y_scale
            # only exact when X_trgt is a constant (e.g. grid case). If not it's a descent approx
            n_z_samples, batch_size, *n_trgt, y_dim = p_y_scale.shape
            p_y_scale = p_y_scale.view(n_z_samples * batch_size, *n_trgt, y_dim)
            p_y_scale = pool_and_replicate_middle(p_y_scale)
            p_y_scale = p_y_scale.view(n_z_samples, batch_size, *n_trgt, y_dim)

        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.PredictiveDistribution(p_y_loc, p_y_scale)

        return p_yCc

    def set_extrapolation(self, min_max):
        """Set the neural process for extrapolation."""
        pass