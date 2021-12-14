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


class NeuralNetwork(nn.Module, abc.ABC):
    """
    Base class for members of the neural process family.

    Notes
    -----
    - when writing size of vectors something like `size=[batch_size,*n_cntxt,y_dim]` means that the
    first dimension is the batch, the last is the target values and everything in the middle are context
    points. We use `*n_cntxt` as it can  be a single flattened dimension or many (for example on the grid).

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    encoded_path : {"latent", "both", "deterministic"}
        Which path(s) to use:
        - `"deterministic"` no latents : the decoder gets a deterministic representation s input.
        - `"latent"` uses latent : the decoder gets a sample latent representation as input.
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder.

    r_dim : int, optional
        Dimension of representations.

    x_transf_dim : int, optional
        Dimension of the encoded X. If `-1` uses `r_dim`. if `None` uses `x_dim`.

    is_heteroskedastic : bool, optional
        Whether the posterior predictive std can depend on the target features. If using in conjuction
        to `NllLNPF`, it might revert to a *CNP model (collapse of latents). If the flag is False, it
        pools all the scale parameters of the posterior distribution. This trick is only exactly
        recovers heteroskedasticity when the set target features are always the same (e.g.
        predicting values on a predefined grid) but is a good approximation even when not.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x^i}_i -> {x_trnsf^i}_i. It should be
        constructable via `XEncoder(x_dim, x_transf_dim)`. `None` uses MLP. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    Decoder : nn.Module, optional
        Decoder module which maps {(x^t, r^t)}_t -> {p_y_suffstat^t}_t. It should be constructable
        via `decoder(x_dim, r_dim, n_out)`. If you have an decoder that maps
        [r;x] -> y you can convert it via `merge_flat_input(Decoder)`. `None` uses MLP. In the
        computational model this corresponds to `g`.
        Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : predict
            with self attention mechanisms (using `X_transf + Y` as input) to have
            coherent predictions (not use in attentive neural process [1] but in
            image transformer [2]).
            - `discard_ith_arg(MLP, 0)` if want the decoding to only depend on r.

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The input to the constructor are currently two values of the same
        shape : `loc` and `scale`, that are preprocessed by `p_y_loc_transformer` and
        `pred_scale_transformer`.

    p_y_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    p_y_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.01.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    [2] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    """



    def __init__(
        self,
        x_dim,
        y_dim,
        r_dim=128,
        x_transf_dim=-1,
        is_heteroskedastic=True,
        XEncoder=None,
        Decoder=None,
        PredictiveDistribution=MultivariateNormalDiag,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=lambda y_scale: 0.01 + 0.99 * F.softplus(y_scale),
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

    def reset_parameters(self):
        weights_init(self)

    @property
    def dflt_Modules(self):
        dflt_Modules = dict()

        dflt_Modules["XEncoder"] = partial(
            MLP, n_hidden_layers=1, hidden_size=self.r_dim
        )

        dflt_Modules["Decoder"] = partial(
            MLP,
            n_hidden_layers=4,
            hidden_size=self.r_dim,
        )


        return dflt_Modules

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