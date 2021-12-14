from utils.ntbks_helpers import get_all_offgrid_datasets, get_img_datasets, add_y_dim, get_covid_datasets, get_all_gp_datasets
import torch
import skorch
from torchvision.utils import make_grid
from PIL import Image
from torchvision import transforms
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.evaluate import test_with_loglikelihood, quantify_global_uncertainty
from npf import NeuralNetwork as NN, CNP, LNP, AttnLNP, ConvCNP, AttnCNP, GridConvCNP
from npf.neuralproc.gbconp import GlobalConvNP
from npf.neuralproc.gbconpgrid import GridGlobalConvNP
from npf.architectures import MLP, merge_flat_input, CNN, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetCustomizedIndcs,
    CovidCntxtTrgtGetter,
    CovidGridCntxtTrgtGetter,
    CovidTimeMasker,
    GetRandomIndcs,
    get_all_indcs,
    GridCntxtTrgtGetter,
    no_masker,
    RandomMasker
)
from npf import CNPFLoss, ELBOLossLNPF
from utils.train import train_models
from npf.utils.helpers import make_abs_conv
from utils.data import cntxt_trgt_collate, context_target_split, cntxt_trgt_collate_covid
from utils.visualize.visualize_imgs import np_input_to_img
import logging
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
from skorch.callbacks import GradientNormClipping

def batch2numpy(x):
    return x.detach().cpu().numpy()

def load_data(namelist=[]):
    path = '/home/xuesongwang/PycharmProject/NPF/data/gp_dataset.hdf5'
    # _, _, test_datasets = get_all_gp_datasets(path)
    _, _, test_datasets = get_all_offgrid_datasets(path)
    if len(namelist) == 0: # does not assign dataset list
        namelist = list(test_datasets.keys())
        return test_datasets, namelist

    datasets = {dataname: test_datasets[dataname] for dataname in namelist}
    return datasets, namelist

def load_img_data(namelist =[]):
    if len(namelist) == 0:  # does not assign dataset list
        namelist = ["MNIST", "SVHN", "CelebA32"]
    _, img_test_datasets = get_img_datasets(namelist)
    return img_test_datasets, namelist

def load_covid(namelist = []):
    if len(namelist) == 0:  # does not assign dataset list
        namelist = ["Covid"]

    img_datasets, img_val_datasets, img_test_datasets = get_covid_datasets(["Covid"],
                                                                            root='/share/scratch/xuesongwang/metadata/Covid',
                                                                            patch_size=104)
    return img_test_datasets, namelist

def define_load_model(testsets, modellist=[], sample_size = 16):
    modeldict = dict()
    if len(modellist) == 0: # does not assign dataset list
        modellist = ['NN', 'CNP', 'NP', 'ANP', 'ACNP', 'ConvCNP', 'GBCoNP', 'ANP_zdim4', 'GBCoNP_zdim4']
    for modelname in modellist:
        if modelname == 'NN':
            R_DIM = 128
            KWARGS = dict(r_dim=R_DIM)
            model_1d = partial(NN,x_dim=1,y_dim=1,**KWARGS)
        elif modelname == 'CNP':
            R_DIM = 128
            KWARGS = dict(
                XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
                Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
                ),
                r_dim=R_DIM,
            )
            model_1d = partial(
                CNP, x_dim=1, y_dim=1,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True),
                **KWARGS,
            )
        elif modelname == 'NP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # number of samples when eval
                XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
                Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
                ),
                r_dim=R_DIM,
            )

            # 1D case
            model_1d = partial(
                LNP, x_dim=1, y_dim=1,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True),**KWARGS )
        elif modelname == 'ANP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_1d = partial(
                AttnLNP, x_dim=1,y_dim=1,
                # z_dim=4,
                z_dim = R_DIM,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
                ),is_self_attn=False,**KWARGS,)
        elif modelname == 'ANP_zdim4':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_1d = partial(
                AttnLNP, x_dim=1,y_dim=1,
                z_dim=4,
                # z_dim = R_DIM,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
                ),is_self_attn=False,**KWARGS,)
        elif modelname == 'ConvCNP':
            R_DIM = 128
            KWARGS = dict( r_dim=R_DIM,Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0),)

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,is_chan_last=True, n_conv_layers=2)

            # off the grid
            model_1d = partial(
                ConvCNP,
                x_dim=1, y_dim=1, Interpolator=SetConv,
                CNN=partial(
                    CNN,
                    Conv=torch.nn.Conv1d,
                    Normalization=torch.nn.BatchNorm1d,
                    n_blocks=5,
                    kernel_size=19,
                    **CNN_KWARGS,
                ),density_induced=64,  **KWARGS,)
        elif modelname == 'GBCoNP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,
                r_dim=R_DIM,
                # z_dim=4,
                z_dim=R_DIM//2,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_1d = partial(
                GlobalConvNP,
                x_dim=1,
                y_dim=1,
                Interpolator=SetConv,
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv1d,
                    Normalization=torch.nn.BatchNorm1d,
                    n_blocks=4,
                    kernel_size=19,
                    **CNN_KWARGS,
                ),
                density_induced=64,  # density of discretization
                **KWARGS,
            )
        elif modelname == 'GBCoNP_zdim4':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,
                r_dim=R_DIM,
                z_dim=4,
                # z_dim=R_DIM//2,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_1d = partial(
                GlobalConvNP,
                x_dim=1,
                y_dim=1,
                Interpolator=SetConv,
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv1d,
                    Normalization=torch.nn.BatchNorm1d,
                    n_blocks=4,
                    kernel_size=19,
                    **CNN_KWARGS,
                ),
                density_induced=64,  # density of discretization
                **KWARGS,
            )
        modeldict[modelname] = model_1d
    # load testing parameter
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/pretrained/",
        device=device,  # use GPU if available
        batch_size=1,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        # seed = 123,
    )


    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(
            contexts_getter=GetRandomIndcs(a=4, b=5), targets_getter=get_all_indcs,
            # contexts_getter=GetCustomizedIndcs(), targets_getter=get_all_indcs,
            is_add_cntxts_to_trgts=False)
    )

    # 1D
    trainers_1d = train_models(
        testsets,
        modeldict,
        test_datasets=testsets,
        iterator_train__collate_fn=get_cntxt_trgt_1d,
        iterator_valid__collate_fn=get_cntxt_trgt_1d,
        max_epochs=100,
        **KWARGS
    )


    return trainers_1d, modellist, modeldict

def define_load_img_model(testsets, modellist=[], sample_size = 16):
    modeldict = dict()
    for name in namelist:
        modeldict[name] = dict()
    if len(modellist) == 0: # does not assign dataset list
        modellist = ['NN', 'CNP', 'NP', 'ANP', 'ACNP', 'ConvCNP', 'ConvNP',  'ANP_zdim4', 'GBCoNP']
    for modelname in modellist:
        if modelname == 'NN':
            R_DIM = 128
            KWARGS = dict(r_dim=R_DIM)
            model_2d = partial(
                NN,
                x_dim=2,
                **KWARGS,
            )
        elif modelname == 'CNP':
            R_DIM = 128
            KWARGS = dict(
                XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
                Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
                ),
                r_dim=R_DIM,
            )
            model_2d = partial(
                CNP,
                x_dim=2,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
                ),
                **KWARGS,
            )  # don't add y_dim yet because depends on data (colored or gray scale)
        elif modelname == 'NP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # number of samples when eval
                XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
                Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
                ),
                r_dim=R_DIM,
            )

            # 1D case
            model_2d = partial(
                LNP,
                x_dim=2,
                XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
                    partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
                ),
                **KWARGS,
            )
        elif modelname == 'ACNP':
            R_DIM = 128
            MODEL_KWARGS = dict(
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
                XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
                Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
                ),
            )

            # 1D case
            model_2d = partial(
                AttnCNP,
                x_dim=2,
                is_self_attn=True,  # no XYEncoder because using self attention
                **MODEL_KWARGS,
            )  # don't add y_dim yet because depends on data (colored or gray scale)
        elif modelname == 'ANP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_2d = partial(
                AttnLNP, x_dim=2,
                z_dim=R_DIM,
                is_self_attn=True, **KWARGS
            )
        elif modelname == 'ANP_zdim4':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_2d = partial(
                AttnLNP, x_dim=2,
                z_dim=4,
                is_self_attn=True, **KWARGS
            )
        elif modelname == 'ConvCNP':
            R_DIM = 128
            MODEL_KWARGS = dict(
                r_dim=R_DIM,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            # off the grid
            model_2d = partial(
                GridConvCNP,
                x_dim=1,  # for gridded conv it's the mask shape
                CNN=partial(
                    CNN,
                    Conv=torch.nn.Conv2d,
                    Normalization=torch.nn.BatchNorm2d,
                    n_blocks=5,
                    kernel_size=9,
                    **CNN_KWARGS,
                ),
                **MODEL_KWARGS,
            )
        elif modelname == 'GBCoNP':
            R_DIM = 128
            MODEL_KWARGS = dict(
                is_q_zCct=True,
                n_z_samples_train=4,
                n_z_samples_test=4,
                r_dim=R_DIM,
                z_dim=R_DIM // 2,
                # z_dim = 4,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_2d = partial(
                GridGlobalConvNP,
                x_dim=1,  # for gridded conv it's the mask shape
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv2d,
                    Normalization=torch.nn.BatchNorm2d,
                    n_blocks=4,
                    kernel_size=9,
                    **CNN_KWARGS,
                ),
                **MODEL_KWARGS,
            )
        elif modelname == 'GBCoNP_zdim4':
            R_DIM = 128
            MODEL_KWARGS = dict(
                is_q_zCct=True,
                n_z_samples_train=4,
                n_z_samples_test=4,
                r_dim=R_DIM,
                # z_dim=R_DIM // 2,
                z_dim = 4,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_2d = partial(
                GridGlobalConvNP,
                x_dim=1,  # for gridded conv it's the mask shape
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv2d,
                    Normalization=torch.nn.BatchNorm2d,
                    n_blocks=4,
                    kernel_size=9,
                    **CNN_KWARGS,
                ),
                **MODEL_KWARGS,
            )
        for name in namelist:
            modeldict[name][modelname] = partial(model_2d, y_dim=testsets[name].shape[0])

    # load testing parameter
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/pretrained/",
        device=device,  # use GPU if available
        batch_size=1,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        # seed=123,
    )

    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0.1, b=0.2), target_masker=no_masker,
        )
    )

    # 1D
    trainers_2d = train_models(
        testsets,
        modeldict,
        test_datasets=testsets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_2d,
        iterator_valid__collate_fn=get_cntxt_trgt_2d,
        max_epochs=50,
        **KWARGS
    )


    return trainers_2d, modellist, modeldict

def define_load_covid_model(testsets, modellist=[], sample_size = 16):
    modeldict = dict()
    if len(modellist) == 0:  # does not assign dataset list
        modellist = ['ANP','ConvNP', 'ANP_zdim4', 'GBCoNP', 'GBCoNP_zdim4']
    for modelname in modellist:
        if modelname == 'ANP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_3d = partial(
                AttnLNP, x_dim=3, y_dim=1,
                z_dim=R_DIM//2,
                is_self_attn=False, **KWARGS
            )
        elif modelname == 'ANP_zdim4':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,  # small number of sampled because Attn is memory intensive
                r_dim=R_DIM,
                attention="transformer",  # multi headed attention with normalization and skip connections
            )

            # 1D case
            model_3d = partial(
                AttnLNP, x_dim=3, y_dim=1,
                z_dim = 4,
                is_self_attn=False, **KWARGS
            )
        elif modelname == 'GBCoNP':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,
                r_dim=R_DIM,
                z_dim=R_DIM // 2,
                # z_dim=4,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_3d = partial(
                GridGlobalConvNP,
                x_dim=1, # for gridded conv it's the mask shape
                y_dim=1,
                Conv=lambda y_dim: make_abs_conv(torch.nn.Conv3d)(
                    y_dim,
                    y_dim,
                    groups=y_dim,
                    kernel_size=[5, 5, 5],
                    padding=[5 // 2, 5 // 2, 5 // 2],
                    bias=False,
                ),
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv3d,
                    Normalization=torch.nn.BatchNorm3d,
                    n_blocks=4,
                    kernel_size=[5, 5, 5],
                    **CNN_KWARGS,
                ),
                **KWARGS,
            )
        elif modelname == 'GBCoNP_zdim4':
            R_DIM = 128
            KWARGS = dict(
                is_q_zCct=True,  # will use NPVI => posterior sampling
                n_z_samples_train=sample_size,
                n_z_samples_test=sample_size,
                r_dim=R_DIM,
                # z_dim=R_DIM // 2,
                z_dim=4,
                Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
                    partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
                ),
            )

            CNN_KWARGS = dict(
                ConvBlock=ResConvBlock,
                is_chan_last=True,  # all computations are done with channel last in our code
                n_conv_layers=2,  # layers per block
            )

            model_3d = partial(
                GridGlobalConvNP,
                x_dim=1, # for gridded conv it's the mask shape
                y_dim=1,
                Conv=lambda y_dim: make_abs_conv(torch.nn.Conv3d)(
                    y_dim,
                    y_dim,
                    groups=y_dim,
                    kernel_size=[5, 5, 5],
                    padding=[5 // 2, 5 // 2, 5 // 2],
                    bias=False,
                ),
                decoder_CNN=partial(
                    CNN,
                    n_channels=R_DIM,
                    Conv=torch.nn.Conv3d,
                    Normalization=torch.nn.BatchNorm3d,
                    n_blocks=4,
                    kernel_size=[5, 5, 5],
                    **CNN_KWARGS,
                ),
                **KWARGS,
            )
        modeldict[modelname] = model_3d
    # load testing parameter
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/pretrained/",
        device=device,  # use GPU if available
        batch_size=1,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        # seed=123,
    )

    get_cntxt_trgt_covid = cntxt_trgt_collate_covid(
        CovidCntxtTrgtGetter(
            context_masker=CovidTimeMasker(timestep=4), target_masker=no_masker,
        )
    )

    # 1D
    trainers_3d = train_models(
        testsets,
        modeldict,
        test_datasets=testsets,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
        iterator_train__collate_fn=get_cntxt_trgt_covid,
        iterator_valid__collate_fn=get_cntxt_trgt_covid,
        max_epochs=50,
        **KWARGS
    )

    return trainers_3d, modellist

def plot_by_dataset(trainers_1d, data, namelist, modellist, num_context =15, num_total = 40):
    import seaborn as sns
    import matplotlib.pyplot as plt
    def generate_z_sample(z_dist, n_z_sample=1):
        z_mean = z_dist.mean
        z_std = z_dist.stddev
        z_range = np.arange(0, 2, 2/(n_z_sample//2))
        z_samples = [z_mean + z_std*i for i in z_range] + [z_mean - z_std*i for i in z_range]
        z_samples = torch.stack(z_samples, dim=0)
        return z_samples

    def sample_with_global_uncertainty(model, modelname, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        X_cntxt = model.x_encoder(X_cntxt)
        X_trgt = model.x_encoder(X_trgt)
        R = model.encode_globally(X_cntxt, Y_cntxt)
        if model.encoded_path in ["latent", "both"]:
            model.n_z_samples = model.n_z_samples_test
            z_samples, z_samples_mean, q_zCc, q_zCct = model.latent_path(X_cntxt, R, X_trgt, Y_trgt)
            z_samples = generate_z_sample(q_zCc, z_samples.shape[0])
            if modelname == 'ConvNP':
                z_samples = z_samples.repeat(1, 1,  R.shape[-2], 1)
        else:
            z_samples, z_samples_mean, q_zCc, q_zCct = None, None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = model.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = model.decode(X_trgt, R_trgt)
        return p_yCc

    sns.set_theme(style="ticks", color_codes=True)

    for col, (dataname, dataset) in enumerate(data.items()):

        fig, axs = plt.subplots(len(modellist), 1,
                                figsize=(30, int(12 * len(modellist))))
        # load data
        #get one batch
        for batch in tqdm(next(iter(trainers_1d.values())).get_iterator(dataset, training=False)):
            x_context = batch[0]['X_cntxt'].to(device)
            y_context = batch[0]['Y_cntxt'].to(device)
            x_target = batch2numpy(batch[0]['X_trgt'][0])
            y_target = batch2numpy(batch[0]['Y_trgt'][0])
            sort_target_index = np.argsort(x_target[:,0])
            x_target = x_target[sort_target_index]
            y_target = y_target[sort_target_index]

            if np.random.rand() >0.5:
                break
        for row, modelname in enumerate(modellist):
            ax = axs if len(modellist) == 1 else axs[row]
            direct = dataname + '/' + modelname + '/run_0'  # look up model by modelname and dataname

            # load model
            # temp = trainers_1d[direct]
            model = trainers_1d[direct].module_.eval().to(device)

            x_min = -1
            x_max = 1

            print("Now with the dataset:", dataname)

            x_all = torch.linspace(x_min, x_max, 200)[None, :, None].to(device)

            dist, sample, z_dist, _ = model(x_context, y_context, x_all.repeat([x_context.shape[0], 1, 1]))

            # dist = sample_with_global_uncertainty(model, modelname, x_context, y_context, x_all.repeat([x_context.shape[0], 1, 1]))
            y_mean = dist.mean[:,0,:,:] # only the first batch
            y_std = dist.stddev[:,0,:,:]
            # Plot context set
            ax.scatter(batch2numpy(x_context[0]), batch2numpy(y_context[0]),
                           label='Context point',
                           s=1200, zorder=2)
            # ax.scatter(batch2numpy(x_target[0]), batch2numpy(y_target[0]), label='Target point',
            #                       marker = 'X', s=800, zorder=2)
            ax.plot(x_target,y_target, linewidth=12, zorder=1, alpha= 1,
                        linestyle='--',color="#6D90F1", label='sampling function')

            # plot prediction
            linecolor = '#F5A081'
            fill_color = '#6D90F1'
            if y_mean.shape[0] != 1: # NP based models
                sample_size = y_mean.shape[0]

            y_sample = batch2numpy(y_mean.squeeze(dim=-1))
            y_std_sample = batch2numpy(y_std.squeeze(dim=-1))

            # y_sample, y_std_sample = latent_sample(x_context, y_context, x_all.repeat([x_context.shape[0], 1, 1]), model, sample_size=sample_size)
            # y_sample = sample_prediction(y_mean.squeeze(), y_std.squeeze(), 1)

            ax.plot(batch2numpy(x_all.squeeze()), y_sample.T, linewidth=6, zorder=1, alpha= 8/sample_size,
                        color=linecolor)

            for sample_i in range(y_mean.shape[0]):
                if y_mean.shape[0] == 1: # CNP based methods, make sure the filled color look alike the NP based methods
                    for _ in range(sample_size):
                        ax.fill_between(batch2numpy(x_all.squeeze()),
                                        y_sample[0] + 2 * y_std_sample[0],
                                        y_sample[0] - 2 * y_std_sample[0],
                                        alpha=2 / (10 * sample_size),
                                        color=fill_color)
                else:
                    ax.fill_between(batch2numpy(x_all.squeeze()),
                                        y_sample[sample_i] + 2 * y_std_sample[sample_i],
                                        y_sample[sample_i] - 2 * y_std_sample[sample_i],
                                        alpha=2/(10*sample_size),
                                        color=fill_color)

            # axs[i].xlabel("Location", size=60)
            ax.set_ylabel("%s" % modelname, size=80)
            if row == 0: # first ax
                ax.set_title(dataname.split('_')[0], {'fontsize': 80})# remove _kernel
            ax.legend(prop={'size': 50}, loc='lower right')
            ax.grid("on", linewidth=3)

            # ax.tick_params(axis="x", labelsize=50)
            [i.set_linewidth(6) for i in ax.spines.values()]  # change subplot border line width
            ax.tick_params(labelsize=50)
        plt.tight_layout()

        fig.savefig('results/imgs/'+dataname + ".png")
        plt.show()

def plot_img_by_dataset(trainers_2d, data, namelist, modellist, n_subplot=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    sns.set_theme(style="ticks", color_codes=True)

    def generate_z_sample(z_dist, n_z_sample=1):
        z_mean = z_dist.mean
        z_std_gt = z_dist.stddev
        rand_index = np.random.randint(z_std_gt.shape[-1])
        z_std = torch.zeros(z_mean.shape).to(device)
        if len(z_std.shape) == 3:
            z_std[:,:,rand_index] = z_std[:,:,rand_index]
        else:
            z_std[:, :, :, rand_index] = z_std[:, :, :, rand_index]
        z_range = np.arange(0, 2, 2(n_z_sample//2))
        z_samples = [z_mean + z_std*(z_range[-1] - i) for i in z_range] + [z_mean - z_std*i for i in z_range]
        z_samples = torch.stack(z_samples, dim=0)
        return z_samples

    def preprocess_image(dataname, dataset):
        imgs = dataset.data
        rand_index = np.random.randint(len(imgs))
        img = imgs[rand_index]
        if dataname == 'MNIST':
            img_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
            img = Image.fromarray(img.numpy(), mode='L')
            img = img_transform(img)
        elif dataname == 'SVHN':
            img_transform = transforms.ToTensor()
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = img_transform(img)
        elif dataname == 'CelebA32': # it's already a ImageFolder
            img = img[0]
        img = img.unsqueeze(0)
        return img, rand_index


    def build_customized_mnist(dataset, dataname, mask_prob):
        img_to_np_input = GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=mask_prob*0.9, b=mask_prob), target_masker=no_masker,
        )
        img, index = preprocess_image(dataname, dataset)
        x_context, y_context, x_target, y_target = img_to_np_input(img)
        return x_context.to(device), y_context.to(device), x_target.to(device), y_target.to(device), index

    def sample_with_global_uncertainty(model, modelname, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        X_cntxt = model.x_encoder(X_cntxt)
        X_trgt = model.x_encoder(X_trgt)
        R = model.encode_globally(X_cntxt, Y_cntxt)
        if model.encoded_path in ["latent", "both"]:
            model.n_z_samples = model.n_z_samples_test
            z_samples, z_samples_mean, q_zCc, q_zCct = model.latent_path(X_cntxt, R, X_trgt, Y_trgt)
            # z_samples = generate_z_sample(q_zCc, 20)
            # if modelname == 'ConvNP':
            #     z_samples = z_samples.repeat(1, 1, R.shape[-3], R.shape[-2], 1)
        else:
            z_samples, z_samples_mean, q_zCc, q_zCct = None, None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = model.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = model.decode(X_trgt, R_trgt)
        return p_yCc

    for col, (dataname, dataset) in enumerate(data.items()):
        fig, axs = plt.subplots(len(modellist),1, figsize=(100, 10*len(modellist)))
        # load data
        # get one batch
        mask_probs = [0.05, 0.1, 0.15, 0.3]
        grid_by_probs = []
        for mask_prob in mask_probs:
            x_context, y_context, x_target, y_target, img_index = build_customized_mnist(dataset, dataname, mask_prob)
            grid_by_model = []
            for row, modelname in enumerate(modellist):
                x_context_cp = x_context.clone()
                y_context_cp = y_context.clone()
                x_target_cp = x_target.clone()
                y_target_cp = y_target.clone()

                print("now with dataset:%s, model:%s"%(dataname, modelname))
                direct = dataname + '/' + modelname + '/run_0'
                model = trainers_2d[direct].module_.eval().to(device)
                masked_image = np_input_to_img(x_context_cp.clone(), y_context_cp.clone(), dataset.shape)
                if 'Co' in modelname:  # ConvCNP
                    x_context_cp = np_input_to_img(x_context_cp, torch.ones(y_context_cp.shape), dataset.shape).bool()
                    x_context_cp = x_context_cp.permute(0, 2, 3, 1).to(device)
                    y_context_cp = masked_image.clone().permute(0, 2, 3, 1).to(device)
                    x_target_cp = np_input_to_img(x_target_cp, torch.ones(y_target_cp.shape), dataset.shape).bool()
                    x_target_cp = x_target_cp.permute(0, 2, 3, 1).to(device)
                    y_target_cp = y_target_cp.permute(0, 2, 1).view(-1, *dataset.shape)
                    y_target_cp = y_target_cp.permute(0, 2, 3, 1).to(device)

                dist = sample_with_global_uncertainty(model, modelname, x_context_cp, y_context_cp, x_target_cp)
                # dist, sample, z_dist, _ = model(x_context_cp, y_context_cp, x_target_cp)
                # if modelname == 'NN' or 'CNP' in modelname:  # plot conditional methods by 3 samples, 9 figures in total
                # [masked_context, predicted_mean, predicted_std]*3 batch
                y_mean = dist.mean[0] # (bs, n_target, y_dim)
                y_std = dist.stddev[0]
                if 'Co' not in modelname:  # ConvCNP
                    y_mean = y_mean.permute(0, 2, 1).view(-1, *dataset.shape)
                    y_std = y_std.permute(0, 2, 1).view(-1, *dataset.shape)
                else:
                    y_mean = y_mean.permute(0, 3, 1, 2)
                    y_std = y_std.permute(0, 3, 1, 2)
                grid_image = torch.cat([masked_image.unsqueeze(0).cpu(), y_mean.unsqueeze(0).cpu(), y_std.unsqueeze(0).cpu()], dim=0)
                grid_image = grid_image.permute(1, 0, 2, 3, 4).contiguous().view(-1, *dataset.shape)
                # (3, *image_shape): (masked_img + mean + std)
                grid_by_model.append(grid_image)
            grid_by_model = torch.stack(grid_by_model, dim=0) # (n_models, 3, *image_shape)
            grid_by_probs.append(grid_by_model)
        grid_by_probs = torch.stack(grid_by_probs, dim=0) #(len(probs), n_models, 3, *image_shape)
        grid_by_model = grid_by_probs.transpose(0, 1) #(n_models,  len(probs), 3, *image_shape)
        for row, modelname in enumerate(modellist):
            ax = axs[row] if len(modellist) !=1 else axs
            img_recover = grid_by_model[row]
            img_recover = img_recover.contiguous().view(-1, *dataset.shape) #(len(probs)*3, *shape)
            img_recover = make_grid(img_recover, nrow=img_recover.shape[0], pad_value=1.).permute(1, 2, 0)
            ax.imshow(img_recover.detach().numpy())
            ax.set_ylabel("%s" % modelname.split('_')[0], size=80)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if row == 0: # first ax
                ax.set_title(dataname, {'fontsize': 120})
        plt.tight_layout()
        fig.savefig('results/imgs/'+dataname + ".png")
        plt.show()
        plt.close(fig)

def plot_covid_by_model(trainers_3d, data, namelist, modellist, n_subplot=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import numpy as np
    sns.set_theme(style="dark", color_codes=True)

    def latent_sample(x_context, y_context, x_all, model, sample_size=5):
        y_mean_list = []
        y_std_list = []
        with torch.no_grad():
            for sample in range(sample_size):
                dist, sample, _, _ = model(x_context, y_context, x_all)
                # get the first n_z_sample & the first sample in the batch
                y_mean = dist.mean[0, 0]
                y_std = dist.stddev[0, 0]
                y_mean_list.append(y_mean.detach())
                y_std_list.append(y_std.detach())
        y_mean_list = torch.stack(y_mean_list)
        y_std_list = torch.stack(y_std_list)
        return y_mean_list, y_std_list

    def img2temporal(raw_image, mean_list, std_list, patch_list):
        """
        raw_image: (total_patch, C, T, H, W)
        mean_list: (n_z, total_patch, C, H, W)
        """
        n_sample = 9
        width_list = [5, 20, 35]
        height_list = [5, 20, 35]
        n_patch = 2
        values = np.zeros((n_patch, n_sample, raw_image.shape[2]))
        prediction_mean = np.zeros((n_patch, n_sample, 2))
        prediction_std = np.zeros((n_patch, n_sample, 2))
        for p, patch in enumerate(patch_list):
            for n in range(n_sample):
                width = width_list[n//3]
                height = height_list[n%3]
                values[p, n, :] = raw_image[patch, 0, :, width, height].cpu().numpy()
                prediction_mean[p, n, 0] = values[p, n, -2]  # today's record as a starting point
                prediction_mean[p, n, 1] = mean_list[:, patch, 0, width, height].cpu().numpy()
                prediction_std[p, n, 1] = std_list[:, patch, 0, width, height].cpu().numpy()
        return values, prediction_mean, prediction_std

    N_ROWS = 3
    N_COL = 4

    for col, (dataname, dataset) in enumerate(data.items()):

        fig, axs = plt.subplots(N_ROWS, N_COL, figsize=(30* N_ROWS, 15* N_COL), \
                                gridspec_kw={'width_ratios': [1, 1, 1, 1], 'height_ratios': [1, 1, 1]})
        # load data
        #get one batch
        for batch in tqdm(next(iter(trainers_3d.values())).get_iterator(dataset, training=False)):
            x_context = batch[0]['X_cntxt'].to(device)
            y_context = batch[0]['Y_cntxt'].to(device)
            x_target = batch[0]['X_trgt'].to(device)
            y_target = batch[0]['Y_trgt'].to(device)
            break

        ######## PLot raw image on the first row
        shape = (4, 40, 40)
        if len(y_target.shape) == 5:  # ConvCNP based methods
            raw_image = y_target.permute(0, 4, 1, 2, 3)
        else:
            raw_image = y_target.permute(0, 2, 1).reshape(-1, 1, *shape)

        ## test to mask the patch
        patch_candidates = [47, 83]
        # raw_image[patch_candidates] = 1
        for t in range(shape[0]):
            groudtruth = make_grid(raw_image[:, :, t, :], nrow=13, normalize=True, pad_value=1).permute(1, 2,
                                                                                                        0)  # plus t in increase contrast
            groudtruth = groudtruth * (0.7 + 0.1 * t)  # amplify contrast
            axs[0, t].imshow(groudtruth[:, :, 0].cpu().detach().numpy(), cmap=plt.get_cmap('Reds'), vmin=0, vmax=1)
            axs[0, t].set_yticklabels([])
            axs[0, t].set_xticklabels([])


        ######### Plot prediction samples and uncertainty on the second and third row
        for row, modelname in enumerate(modellist):
            print("now with dataset:%s, model:%s" % (dataname, modelname))
            direct = dataname + '/' + modelname + '/run_0'
            model = trainers_3d[direct].module_.eval().to(device)

            x_context_cp = x_context.clone()
            y_context_cp = y_context.clone()
            x_target_cp = x_target.clone()
            y_target_cp = y_target.clone()

            if 'Co' in modelname:  # ConvCNP
                x_context_cp = torch.ones(dataset.shape).bool().to(device)
                x_context_cp[:, -1, :] = False
                x_context_cp = x_context_cp.repeat(y_context_cp.shape[0], 1, 1, 1).unsqueeze(dim=-1)
                x_target_cp = torch.ones(dataset.shape).bool().to(device)
                x_target_cp = x_target_cp.repeat(y_target_cp.shape[0], 1, 1, 1).unsqueeze(dim=-1)
                y_context_cp = y_context_cp.contiguous().view(-1, shape[0]-1, shape[1], shape[2], 1)
                y_context_cp_mask = torch.zeros(y_context_cp.shape[0], 1, shape[1], shape[2], 1).to(device)
                y_context_cp = torch.cat([y_context_cp, y_context_cp_mask], dim=1)
                y_target_cp = y_target_cp.contiguous().view(-1, shape[0], shape[1], shape[2], 1)

            # plot the whole country, considering the capacity of the machine, 1 patch at a time
            mean_list = []
            std_list = []

            for i in range(x_context.shape[0]):
                print("patch:", i)
                mean_sample, std_sample = latent_sample(x_context_cp[[i]], y_context_cp[[i]], x_target_cp[[i]],
                                                        model, sample_size=1)
                n_z = mean_sample.shape[0]

                # convert from (n_target, c) to img_size
                if len(y_target.shape) != 5:
                    mean_sample = mean_sample.reshape((n_z, *shape, 1))
                    std_sample = std_sample.reshape((n_z, *shape, 1))
                # only use the prediction img, the last timestep
                # (n_z, shape[1], shape[2], 1)
                mean_sample = mean_sample[:, -1, :]
                std_sample = std_sample[:, -1, :]

                mean_list.append(mean_sample.detach())  # detach to save memory
                std_list.append(std_sample.detach())
            # (n_patch, n_z, shape[1], shape[2], 1)
            mean_list = torch.stack(mean_list, dim=0).permute(1, 0, 4, 2, 3)
            std_list = torch.stack(std_list, dim=0).permute(1, 0, 4, 2, 3)
            # make grids for each sample
            for j in range(n_z):
                mean_sample = make_grid(mean_list[j], nrow=13, pad_value=1).permute(1, 2, 0)
                std_sample = make_grid(std_list[j], nrow=13, pad_value=1).permute(1, 2, 0)
                axs[1, 2*row].imshow(mean_sample[:, :, 0].cpu().detach().numpy(), cmap=plt.get_cmap('Reds'))
                axs[1, 2*row+1].imshow(std_sample[:, :, 0].cpu().detach().numpy(), cmap=plt.get_cmap('Reds'))
            [ax.set_yticklabels([]) for ax in axs[1]]
            [ax.set_xticklabels([]) for ax in axs[1]]
            # ######### Plot prediction samples and uncertainty on the second and third row
            scatterdots, pred_mean, pred_std = img2temporal(raw_image, mean_list, std_list, patch_candidates)
            n_patch = scatterdots.shape[0]
            n_sample = scatterdots.shape[1]
            step = np.array([-7, -3, 0, 7])
            color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for j in range(n_patch):
                scatter_x = np.repeat(step[None, :], n_sample, axis=0).reshape(-1)
                scatter_y = scatterdots[j].reshape(-1)
                # scatter ground truth
                axs[2, 2*row+j].scatter(scatter_x, scatter_y, s=1000, zorder=2.5)
                # scatter predictions
                # axs[row + 1, j + 2].scatter(np.ones(n_sample)*7, pred_mean[j][:, 1], marker='*', s=2000, zorder=2)
                # plot between
                for n in range(n_sample):
                    # plot the ground truth values
                    axs[2, 2*row+j].plot(step, scatterdots[j, n], linewidth=20, color = color_list[n])
                    # scatter the ground truth values
                    # plot prediction
                    axs[2, 2*row+j].plot(np.array([0, 7]), pred_mean[j, n], linestyle='--', linewidth=20, color=color_list[n],zorder=1)


                    axs[2, 2*row+j].fill_between([0, 7], pred_mean[j, n] - 1.5 * pred_std[j, n],
                                           pred_mean[j, n] + 1.5 * pred_std[j, n],
                                           color=color_list[n],
                                           alpha=0.3,
                                           zorder=1)
                axs[2, 2*row+j].set_ylim(top=8)
                axs[2, 2*row+j].grid("on", linewidth=3)
                axs[2, 2*row+j].tick_params(labelsize=50)
        plt.tight_layout()
        fig.savefig('results/imgs/'+dataname + ".png")
        plt.show()
        plt.close(fig)

def compute_likelihood(mean, var, y_target, issum=True):
    from torch.distributions import Normal
    dist = Normal(loc=mean, scale=var)

    if mean.shape[0] != y_target.shape[0]:  # z_sample
        y_target = y_target.expand(mean.size())
    log_prob = dist.log_prob(y_target)
    if len(log_prob) > 3: # Covid
        loss = torch.mean(log_prob, dim=[1,2,3])[:, 0]
    else:
        loss = torch.mean(log_prob, dim=-2)[:, 0]
    n_row = int(np.sqrt(loss.shape[0]))
    loss = batch2numpy(loss.reshape((n_row, n_row)))
    return loss

def visualize_global_uncertainty(trainers_1d, data, namelist, modellist, n_subplot=1, sample_size=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from torchvision.utils import make_grid
    sns.set_theme(style="ticks", color_codes=True)

    def generate_z_sample(z_dist, n_z_sample=1):
        z_mean = z_dist.mean
        z_std = z_dist.stddev
        bs, *data_shape, dim = z_mean.shape
        z_grid = z_mean[None, None, :].repeat(n_z_sample, n_z_sample, 1, 1, 1)
        # rand_index = np.random.choice(np.arange(0, dim, 1), 2, replace=False)  # non-replacement shuffle
        rand_index = [0, 2]
        print("index combo:", rand_index)
        rand_index = np.random.randint(0, dim, 2)
        x = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample), loc=z_mean[0,0,rand_index[0]].detach().cpu().numpy(),
                     scale=15*z_std[0, 0, rand_index[0]].detach().cpu().numpy())
        y = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample), loc=z_mean[0,0,rand_index[1]].detach().cpu().numpy(),
                     scale=15*z_std[0, 0, rand_index[1]].detach().cpu().numpy())
        l_x = x.shape[0]
        l_y = y.shape[0]
        x_grid = np.repeat(x, l_y).reshape(-1, 1)
        y_grid = np.tile(y, l_x).reshape(-1, 1)
        _z = np.concatenate([x_grid, y_grid], axis=1)
        for i in range(n_z_sample):
            for j in range(n_z_sample):
                z_grid[i, j, :, :, rand_index[0]] = _z[i*n_z_sample+j, 0]
                z_grid[i, j, :, :, rand_index[1]] = _z[i * n_z_sample + j, 1]
        z_samples = z_grid.view(n_z_sample*n_z_sample, bs, *data_shape, dim)
        return z_samples

    def sample_with_global_uncertainty(model, modelname, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None, sample_size=1):
        X_cntxt = model.x_encoder(X_cntxt)
        X_trgt = model.x_encoder(X_trgt)
        R = model.encode_globally(X_cntxt, Y_cntxt)


        if model.encoded_path in ["latent", "both"]:
            model.n_z_samples = model.n_z_samples_test
            z_samples, z_samples_mean, q_zCc, q_zCct = model.latent_path(X_cntxt, R, X_trgt, Y_trgt)
            temp1 = q_zCc.mean.detach().cpu().numpy()
            temp2 = q_zCc.stddev.detach().cpu().numpy()
            z_samples = generate_z_sample(q_zCc, sample_size)
            if 'Co' in modelname:
                z_samples = z_samples.repeat(1, 1, R.shape[-2], 1)
        else:
            z_samples, z_samples_mean, q_zCc, q_zCct = None, None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = model.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = model.decode(X_trgt, R_trgt)
        return p_yCc

    for col, (dataname, dataset) in enumerate(data.items()):
        linecolor = '#F5A081'
        fill_color = '#6D90F1'
        n_row = sample_size
        n_col = sample_size
        fig, axs = plt.subplots(n_row, n_col, figsize=(50, 20))
        # load data
        # get one batch
        for batch in tqdm(next(iter(trainers_1d.values())).get_iterator(dataset, training=True)):
            x_context = batch[0]['X_cntxt'].to(device)
            y_context = batch[0]['Y_cntxt'].to(device)
            x_target = batch[0]['X_trgt'].to(device)
            y_target = batch[0]['Y_trgt'].to(device)
            if np.random.rand() > 0.5:
                break

        for row, modelname in enumerate(modellist):
            x_context_cp = x_context.clone()
            y_context_cp = y_context.clone()
            x_target_cp = x_target.clone()
            y_target_cp = y_target.clone()

            print("now with dataset:%s, model:%s" % (dataname, modelname))
            direct = dataname + '/' + modelname + '/run_0'
            model = trainers_1d[direct].module_.eval().to(device)

            dist = sample_with_global_uncertainty(model, modelname, x_context_cp, y_context_cp, x_target_cp, sample_size=sample_size)
            # dist, sample, z_dist, _ = model(x_context_cp, y_context_cp, x_target_cp)
            # [masked_context, predicted_mean*7, predicted_std]
            y_mean = dist.mean[:, 0, :, :]  # (n_z_sample, n_target, y_dim)
            y_std = dist.stddev[:, 0, :, :]

            loglikelihood = compute_likelihood(y_mean, y_std, y_target_cp, issum=False)
            for i in range(n_row):
                for j in range(n_col):
                    ax = axs[i, j]
                    ax.scatter(batch2numpy(x_context_cp[0]), batch2numpy(y_context_cp[0]),
                       s=500, zorder=2)

                    ax.plot(batch2numpy(x_target_cp[0].squeeze()), batch2numpy(y_mean[i*n_row + j]), linewidth=10, zorder=1,
                            color=linecolor)
                    ax.set_xticks([]) # remove xtick, y_tick
                    ax.set_yticks([])
                    [i.set_linewidth(10) for i in ax.spines.values()] # change subplot border line width

        plt.tight_layout()
        fig.savefig('results/' + dataname + ".png")
        plt.show()
        plt.close(fig)

        # draw log-likelihood heatmap
        fig, axs = plt.subplots(figsize=(50, 20))
        sns.set(font_scale=5)  # font size 2
        res = sns.heatmap(loglikelihood, annot=True,
                          annot_kws={"size": 50},
                          fmt=".2f")
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=50)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=50)
        fig.savefig('results/' + dataname + "_heatmap.png")

def visualize_img_global_uncertainty(trainers_2d, data, namelist, modellist, n_subplot=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    sns.set_theme(style="ticks", color_codes=True)

    def generate_z_sample(z_dist, n_z_sample=1):
        z_mean = z_dist.mean
        z_std = z_dist.stddev
        bs, *data_shape, dim = z_mean.shape
        # rand_index = np.random.choice(np.arange(0, dim, 1), 2, replace=False)  # non-replacement shuffle
        rand_index = [2, 3]
        print("index combo:", rand_index)
        z_grid = z_mean[None, None, :].repeat(n_z_sample, n_z_sample, 1, *data_shape, 1)
        if len(data_shape) >1:
            x = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, 0, rand_index[0]].detach().cpu().numpy(),
                         scale=12*z_std[0, 0, 0, rand_index[0]].detach().cpu().numpy())
            y = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, 0, rand_index[1]].detach().cpu().numpy(),
                         scale=12*z_std[0, 0, 0, rand_index[1]].detach().cpu().numpy())
        else:
            x = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample), loc=z_mean[0,0,rand_index[0]].detach().cpu().numpy(),
                         scale=40*z_std[0, 0, rand_index[0]].detach().cpu().numpy())
            y = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample), loc=z_mean[0,0,rand_index[1]].detach().cpu().numpy(),
                         scale=40*z_std[0, 0, rand_index[1]].detach().cpu().numpy())
        l_x = x.shape[0]
        l_y = y.shape[0]
        x_grid = np.repeat(x, l_y).reshape(-1, 1)
        y_grid = np.tile(y, l_x).reshape(-1, 1)
        _z = np.concatenate([x_grid, y_grid], axis=1)
        for i in range(n_z_sample):
            for j in range(n_z_sample):
                if len(data_shape) > 1: # image input
                    z_grid[i, j, :, :, :, rand_index[0]] = _z[i * n_z_sample + j, 0]
                    z_grid[i, j, :, :, :, rand_index[1]] = _z[i * n_z_sample + j, 1]
                else:
                    z_grid[i, j, :, :, rand_index[0]] = _z[i*n_z_sample+j, 0]
                    z_grid[i, j, :, :, rand_index[1]] = _z[i * n_z_sample + j, 1]
        z_samples = z_grid.view(n_z_sample*n_z_sample, bs, *data_shape, dim)
        return z_samples, rand_index

    def sample_with_global_uncertainty(model, modelname, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        X_cntxt = model.x_encoder(X_cntxt)
        X_trgt = model.x_encoder(X_trgt)
        R = model.encode_globally(X_cntxt, Y_cntxt)


        if model.encoded_path in ["latent", "both"]:
            model.n_z_samples = model.n_z_samples_test
            z_samples, z_samples_mean, q_zCc, q_zCct = model.latent_path(X_cntxt, R, X_trgt, Y_trgt)
            temp1 = q_zCc.mean.detach().cpu().numpy()
            temp2 = q_zCc.stddev.detach().cpu().numpy()
            z_samples, zdim_index = generate_z_sample(q_zCc, 10)
            if 'Co' in modelname:
                z_samples = z_samples.repeat(1, 1, R.shape[-3], R.shape[-2], 1)
        else:
            z_samples, z_samples_mean, q_zCc, q_zCct = None, None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = model.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = model.decode(X_trgt, R_trgt)
        return p_yCc, zdim_index

    def preprocess_image(dataname, dataset):
        if dataname == 'MNIST':
            digit = 5
            indices = dataset.targets == digit
            imgs = dataset.data[indices]
            rand_index = np.random.randint(len(imgs))
            # rand_index = 180
            img = imgs[rand_index]
            img_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
            img = Image.fromarray(img.numpy(), mode='L')
            img = img_transform(img)
        elif dataname == 'SVHN':
            digit = ''
            imgs = dataset.data
            rand_index = np.random.randint(len(imgs))
            img = imgs[rand_index]
            img_transform = transforms.ToTensor()
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = img_transform(img)
        elif dataname == 'CelebA32':  # it's already a ImageFolder
            digit = ''
            imgs = dataset.data
            rand_index = np.random.randint(len(imgs))
            img = imgs[rand_index]
            img = img[0]
        img = img.unsqueeze(0)
        return img, rand_index, digit

    def get_random_image(dataname, dataset):
        img_to_np_input = GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0.1, b=0.3), target_masker=no_masker,
        )
        img, rand_index, digit = preprocess_image(dataname, dataset)
        x_context, y_context, x_target, y_target = img_to_np_input(img)
        return x_context.to(device), y_context.to(device), x_target.to(device), y_target.to(device), rand_index, digit

    for col, (dataname, dataset) in enumerate(data.items()):
        fig, axs = plt.subplots(len(modellist), 1, figsize=(20, 20 * len(modellist)))
        # load data
        # get one batch
        x_context, y_context, x_target, y_target, img_index, digit = get_random_image(dataname, dataset)

        for row, modelname in enumerate(modellist):
            x_context_cp = x_context.clone()
            y_context_cp = y_context.clone()
            x_target_cp = x_target.clone()
            y_target_cp = y_target.clone()

            ax = axs if len(modellist) == 1 else axs[row]
            print("now with dataset:%s, model:%s" % (dataname, modelname))
            direct = dataname + '/' + modelname + '/run_0'
            model = trainers_2d[direct].module_.eval().to(device)
            masked_image = np_input_to_img(x_context_cp.clone(), y_context_cp.clone(), dataset.shape)
            if 'Co' in modelname:  # ConvCNP
                x_context_cp = np_input_to_img(x_context_cp, torch.ones(y_context_cp.shape), dataset.shape).bool()
                x_context_cp = x_context_cp.permute(0, 2, 3, 1).to(device)
                y_context_cp = masked_image.clone().permute(0, 2, 3, 1).to(device)
                x_target_cp = np_input_to_img(x_target_cp, torch.ones(y_target_cp.shape), dataset.shape).bool()
                x_target_cp = x_target_cp.permute(0, 2, 3, 1).to(device)
                y_target_cp = y_target_cp.permute(0, 2, 1).view(-1, *dataset.shape)
                y_target_cp = y_target_cp.permute(0, 2, 3, 1).to(device)

            dist, zdim_index = sample_with_global_uncertainty(model, modelname, x_context_cp, y_context_cp, x_target_cp)
            # dist, sample, z_dist, _ = model(x_context_cp, y_context_cp, x_target_cp)
            # [masked_context, predicted_mean*7, predicted_std]
            y_mean = dist.mean[:, 0, :, :]  # (n_z_sample, n_target, y_dim)
            y_std = dist.stddev[:, 0, :, :]

            loglikelihood = compute_likelihood(y_mean, y_std, y_target_cp, issum=False)
            if 'Co' not in modelname:  # ConvCNP
                y_mean = y_mean.permute(0, 2, 1).view(-1, *dataset.shape)
                y_std = y_std.permute(0, 2, 1).view(-1, *dataset.shape)
            else:
                y_mean = y_mean.permute(0, 3, 1, 2)
                y_std = y_std.permute(0, 3, 1, 2)

            # only the last sample comes with std
            grid_image = y_mean.cpu()

            # replace the first element with the masked image
            grid_image[0] = masked_image[0]

            img_recover = make_grid(grid_image, nrow=10, pad_value=1.).permute(1, 2, 0)
            ax.imshow(img_recover.detach().numpy())
            ax.set_ylabel("%s" % modelname.split('_')[0], size=50)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if row == 0:  # first ax
                ax.set_title(dataname, {'fontsize': 50})
        plt.tight_layout()
        fig.savefig('results/'+ str(digit) +'_img'+str(img_index) + 'zdim'+str(zdim_index[0]) +str(zdim_index[1])+".png")
        plt.show()
        plt.close(fig)

        # draw log-likelihood heatmap
        fig, axs = plt.subplots(figsize=(50, 20))
        sns.set(font_scale=5)  # font size 2
        res = sns.heatmap(loglikelihood, annot=True,
                          annot_kws={"size": 50},
                          fmt=".2f")
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=50)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=50)
        fig.savefig('results/' + dataname + "_heatmap.png")

def visualize_covid_global_uncertainty(trainers_3d, data, namelist, modellist, n_subplot=1, sample_size=10):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    sns.set_theme(style="ticks", color_codes=True)

    def generate_z_sample(z_dist, n_z_sample=1):
        z_mean = z_dist.mean
        z_std = z_dist.stddev
        bs, *data_shape, dim = z_mean.shape
        # rand_index = np.random.choice(np.arange(0, dim, 1), 2, replace=False)  # non-replacement shuffle
        rand_index = [0, 1]
        print("index combo:", rand_index)
        z_grid = z_mean[None, None, :].repeat(n_z_sample, n_z_sample, 1, *data_shape, 1)
        if len(data_shape) > 1:
            x = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, 0, 0, rand_index[0]].detach().cpu().numpy(),
                         scale=15 * z_std[0, 0, 0, 0, rand_index[0]].detach().cpu().numpy())
            y = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, 0, 0, rand_index[1]].detach().cpu().numpy(),
                         scale=15* z_std[0, 0, 0, 0, rand_index[1]].detach().cpu().numpy())
        else:
            x = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, rand_index[0]].detach().cpu().numpy(),
                         scale=100 * z_std[0, 0, rand_index[0]].detach().cpu().numpy())
            y = norm.ppf(np.linspace(0.05, 0.95, num=n_z_sample),
                         loc=z_mean[0, 0, rand_index[1]].detach().cpu().numpy(),
                         scale=100 * z_std[0, 0, rand_index[1]].detach().cpu().numpy())
        l_x = x.shape[0]
        l_y = y.shape[0]
        x_grid = np.repeat(x, l_y).reshape(-1, 1)
        y_grid = np.tile(y, l_x).reshape(-1, 1)
        _z = np.concatenate([x_grid, y_grid], axis=1)
        for i in range(n_z_sample):
            for j in range(n_z_sample):
                if len(data_shape) > 1:  # image input
                    z_grid[i, j, :, :, :, :, rand_index[0]] = _z[i * n_z_sample + j, 0]
                    z_grid[i, j, :, :, :, :, rand_index[1]] = _z[i * n_z_sample + j, 1]
                else:
                    z_grid[i, j, :, :, rand_index[0]] = _z[i * n_z_sample + j, 0]
                    z_grid[i, j, :, :, rand_index[1]] = _z[i * n_z_sample + j, 1]
        z_samples = z_grid.view(n_z_sample * n_z_sample, bs, *data_shape, dim)
        return z_samples, rand_index

    def sample_with_global_uncertainty(model, modelname, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None,  sample_size=1):
        X_cntxt = model.x_encoder(X_cntxt)
        X_trgt = model.x_encoder(X_trgt)
        R = model.encode_globally(X_cntxt, Y_cntxt)

        if model.encoded_path in ["latent", "both"]:
            model.n_z_samples = model.n_z_samples_test
            z_samples, z_samples_mean, q_zCc, q_zCct = model.latent_path(X_cntxt, R, X_trgt, Y_trgt)
            temp1 = q_zCc.mean.detach().cpu().numpy()
            temp2 = q_zCc.stddev.detach().cpu().numpy()
            z_samples, zdim_index = generate_z_sample(q_zCc, sample_size)
            if 'Co' in modelname:
                z_samples = z_samples.repeat(1, 1, R.shape[-4], R.shape[-3], R.shape[-2], 1)
        else:
            z_samples, z_samples_mean, q_zCc, q_zCct = None, None, None, None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = model.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = model.decode(X_trgt, R_trgt)
        return p_yCc, zdim_index

    def preprocess_image(dataname, dataset):
        if dataname == 'MNIST':
            digit = 6
            indices = dataset.targets == digit
            imgs = dataset.data[indices]
            rand_index = np.random.randint(len(imgs))
            img = imgs[rand_index]
            img_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
            img = Image.fromarray(img.numpy(), mode='L')
            img = img_transform(img)
        elif dataname == 'SVHN':
            digit = ''
            imgs = dataset.data
            rand_index = np.random.randint(len(imgs))
            img = imgs[rand_index]
            img_transform = transforms.ToTensor()
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = img_transform(img)
        elif dataname == 'CelebA32':  # it's already a ImageFolder
            digit = ''
            imgs = dataset.data
            rand_index = np.random.randint(len(imgs))
            img = imgs[rand_index]
            img = img[0]
        img = img.unsqueeze(0)
        return img, rand_index, digit



    for col, (dataname, dataset) in enumerate(data.items()):
        fig, axs = plt.subplots(1, len(modellist), figsize=(20*len(modellist), 20))
        # load data
        # get one batch
        for batch in tqdm(next(iter(trainers_3d.values())).get_iterator(dataset, training=False)):
            x_context = batch[0]['X_cntxt'].to(device)
            y_context = batch[0]['Y_cntxt'].to(device)
            x_target = batch[0]['X_trgt'].to(device)
            y_target = batch[0]['Y_trgt'].to(device)
            break
        # select the candidate cells
        patch_candidates = [83] #47, 83
        x_context = x_context[patch_candidates]
        y_context = y_context[patch_candidates]
        x_target = x_target[patch_candidates]
        y_target = y_target[patch_candidates]

        shape = (4, 40, 40)
        for row, modelname in enumerate(modellist):
            x_context_cp = x_context.clone()
            y_context_cp = y_context.clone()
            x_target_cp = x_target.clone()
            y_target_cp = y_target.clone()

            ax = axs if len(modellist) == 1 else axs[row]
            print("now with dataset:%s, model:%s" % (dataname, modelname))
            direct = dataname + '/' + modelname + '/run_0'
            model = trainers_3d[direct].module_.eval().to(device)

            if 'Co' in modelname:  # ConvCNP
                x_context_cp = torch.ones(dataset.shape).bool().to(device)
                x_context_cp[:, -1, :] = False
                x_context_cp = x_context_cp.repeat(y_context_cp.shape[0], 1, 1, 1).unsqueeze(dim=-1)
                x_target_cp = torch.ones(dataset.shape).bool().to(device)
                x_target_cp = x_target_cp.repeat(y_target_cp.shape[0], 1, 1, 1).unsqueeze(dim=-1)
                y_context_cp = y_context_cp.contiguous().view(-1, shape[0] - 1, shape[1], shape[2], 1)
                y_context_cp_mask = torch.zeros(y_context_cp.shape[0], 1, shape[1], shape[2], 1).to(device)
                y_context_cp = torch.cat([y_context_cp, y_context_cp_mask], dim=1)
                y_target_cp = y_target_cp.contiguous().view(-1, shape[0], shape[1], shape[2], 1)


            dist, zdim_index = sample_with_global_uncertainty(model, modelname, x_context_cp, y_context_cp, x_target_cp, sample_size=sample_size)
            # dist, sample, z_dist, _ = model(x_context_cp, y_context_cp, x_target_cp)
            # [masked_context, predicted_mean*7, predicted_std]
            y_mean = dist.mean[:, 0, :, :]  # (n_z_sample, n_target, y_dim)
            y_std = dist.stddev[:, 0, :, :]

            loglikelihood = compute_likelihood(y_mean, y_std, y_target_cp, issum=False)
            if 'Co' not in modelname:  # ConvCNP
                y_mean = y_mean.permute(0, 2, 1).view(-1, *dataset.shape)
                y_std = y_std.permute(0, 2, 1).view(-1, *dataset.shape)
            else:
                y_mean = y_mean.permute(0, 4, 1, 2, 3)
                y_std = y_std.permute(0, 4, 1, 2, 3)
            # only the last sample comes with std
            grid_image = y_std.cpu()
            grid_image = grid_image[:,0,[-1],:,:]# only use the last time step
            img_recover = make_grid(grid_image, nrow=sample_size, pad_value=1.).permute(1, 2, 0)
            ax.imshow(img_recover[:, :, 0].detach().numpy(), cmap=plt.get_cmap('Reds'))
            if 'GB' not in modelname:
                ax.set_ylabel("%s" % modelname.split('_')[0], size=50)
            else:
                ax.set_ylabel("MLNP", size=50)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if row == 0:  # first ax
                ax.set_title(dataname, {'fontsize': 50})
        plt.tight_layout()
        fig.savefig('results/'+dataname+str(patch_candidates[0])+ ".png")
        plt.show()
        plt.close(fig)

        # draw log-likelihood heatmap
        fig, axs = plt.subplots(figsize=(50, 20))
        sns.set(font_scale=5)  # font size 2
        res = sns.heatmap(loglikelihood, annot=True,
                          annot_kws={"size": 50},
                          fmt=".2f")
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=50)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=50)
        fig.savefig('results/' + dataname + "_heatmap.png")


def gp_demo_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    sns.set_theme(style="whitegrid", color_codes=True)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
    linecolor = '#F5A081'
    fill_color = '#6D90F1'
    np.random.seed(1)
    X = np.atleast_2d([1., 3., 5., 7., 8.]).T

    # Observations
    y = (1.6* np.sin(X)).ravel()

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T

    # Instantiate a Gaussian Process model

    # kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    kernel2 = ExpSineSquared(length_scale=1.0, periodicity = 1, length_scale_bounds=(1e-2, 10),periodicity_bounds = (0.5, 1.5))
    kernel1 = ExpSineSquared(length_scale=1.0, periodicity = 3, length_scale_bounds=(1e-2, 10),periodicity_bounds = (2, 4))
    gp = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=9)
    gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    gp2.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    y_sample =gp.sample_y(x, n_samples=1)

    y_pred2, sigma2 = gp2.predict(x, return_std=True)
    y_sample2 = gp2.sample_y(x, n_samples=1)
    # Plot the function, the prediction and the 95% confidence interval based on
    #
    fig, ax = plt.subplots(2, 1, figsize=(15,10))


    shuffled_index = np.random.permutation(np.arange(x.shape[0]))
    target_x = x[shuffled_index[:3]]
    target_y = y_sample[shuffled_index[:3]][:,0]



    ax[0].fill_between(x[:,0],
             y_pred - 1.9600 * sigma,
             y_pred + 1.9600 * sigma,
             alpha=.3, color=fill_color, label='Uncertainty')

    ax[1].fill_between(x[:, 0],
                     y_pred2 - 1.9600 * sigma,
                     y_pred2 + 1.9600 * sigma,
                     alpha=.3, color=linecolor, label='Uncertainty')

    # for i in range(y_sample.shape[1]):
    #     plt.plot(x, y_sample[:,i], color=linecolor, linestyle='--', linewidth=3,
    #              label='Function sample %d'%(i+1), zorder=1)
    #     break
    target_y = (1.6 * np.sin(target_x)).ravel()
    ax[0].scatter(target_x, target_y, color='magenta', marker='X', s=500, zorder=2.5, label='Target Set')
    ax[1].scatter(target_x, target_y, color='magenta', marker='X', s=500, zorder=2.5, label='Target Set')
    ax[0].plot(x, y_pred,  linestyle='-', color=fill_color, linewidth=6, label='Average Prediction')
    ax[1].plot(x, y_pred2, linestyle='-', color=linecolor, linewidth=6, label='Average Prediction')
    ax[0].plot(X, y, 'r.', markersize=30, zorder=2.5, label='Context Set')
    ax[1].plot(X, y, 'r.', markersize=30, zorder=2.5, label='Context Set')

    def ax_postprocessing(ax):
        ax.legend(prop={'size': 15}, ncol=4, loc='upper left')
        ax.set_ylim(-4, 4)
        ax.grid("on", linewidth=2)
        ax.tick_params(labelsize=20)
        [i.set_linewidth(1) for i in ax.spines.values()]
    ax_postprocessing(ax[0])
    ax_postprocessing(ax[1])
    plt.savefig("demo1.png")
    plt.show()

def z_prior_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import math

    def py_bivariate_normal_pdf(domain, mean, variance):
        X = [[-mean + x * variance for x in range(int((-domain + mean) // variance),
                                                  int((domain + mean) // variance) + 1)]
             for _ in range(int((-domain + mean) // variance),
                            int((domain + mean) // variance) + 1)]
        Y = [*map(list, zip(*X))]
        R = [[math.sqrt(a ** 2 + b ** 2) for a, b in zip(c, d)] for c, d in zip(X, Y)]
        Z = [[(1. / math.sqrt(2 * math.pi)) * math.exp(-.5 * r ** 2) for r in r_sub] for r_sub in R]
        X = [*map(lambda a: [b + mean for b in a], X)]
        Y = [*map(lambda a: [b + mean for b in a], Y)]
        return np.array(X), np.array(Y), np.array(Z)

    def plt_plot_bivariate_normal_pdf(x, y, z):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.axis('off')
        plt.savefig('z_prior.png')
        plt.show()

    plt_plot_bivariate_normal_pdf(*py_bivariate_normal_pdf(6, 4, .25))

def gb_cntxt_correlation(testsets, modeldict, device):
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/pretrained/",
        device=device,  # use GPU if available
        batch_size=1,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        # seed=123,
    )

    # initialize the dict
    df_gb = {}
    for name in namelist:
        df_gb[name] = []


    for context_size in range(0, 10, 1):
        print("context size:%d"%context_size)
        get_cntxt_trgt_1d = cntxt_trgt_collate(
            CntxtTrgtGetter(
                contexts_getter=GetRandomIndcs(a=context_size, b=context_size+1), targets_getter=get_all_indcs,
                # contexts_getter=GetCustomizedIndcs(), targets_getter=get_all_indcs,
                is_add_cntxts_to_trgts=False)
        )

        # 1D
        trainers_1d = train_models(
            testsets,
            modeldict,
            test_datasets=testsets,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            max_epochs=100,
            **KWARGS
        )

        for k, trainer in trainers_1d.items():
            kernel = k.split('/')[0]
            modelname = k.split('/')[1]
            print("testing on:", kernel)
            df = quantify_global_uncertainty(testsets[kernel], trainer)
            df_gb[kernel].append(df.values[0,1]) # only need stddev

    df_gb = pd.DataFrame(df_gb)
    df_gb.to_csv("%s_global_uncertainty.csv"%(modellist[0]))

def gb_cntxt_img_correlation(testsets, modeldict, device):
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=ELBOLossLNPF,  # NPML
        chckpnt_dirname="results/pretrained/",
        device=device,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=32,  # smaller batch because multiple samples
        callbacks=[
            GradientNormClipping(gradient_clip_value=1)
        ],  # clipping gradients can stabilize training
    )


    # initialize the dict
    df_gb = {}
    for name in namelist:
        df_gb[name] = []

    for context_prop in np.arange(0, 0.1, 0.01):
        print("context proportion:%.3f" % context_prop)
        if 'Co' in modellist[0]:
            get_cntxt_trgt_2d = cntxt_trgt_collate(
                GridCntxtTrgtGetter(
                    context_masker=RandomMasker(a=context_prop, b=context_prop+0.01), target_masker=no_masker,
                ),
                is_return_masks=True,  # will be using grid conv CNP => can work directly with mask
            )
        else:
            get_cntxt_trgt_2d = cntxt_trgt_collate(
                GridCntxtTrgtGetter(
                    context_masker=RandomMasker(a=context_prop, b=context_prop+0.01), target_masker=no_masker,
                )
            )
        # 2D
        trainers_2d = train_models(
            testsets,
            modeldict,
            test_datasets=testsets,
            train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            max_epochs=5,
            **KWARGS
        )

        for k, trainer in trainers_2d.items():
            kernel = k.split('/')[0]
            modelname = k.split('/')[1]
            print("testing on:", kernel)
            df = quantify_global_uncertainty(testsets[kernel], trainer)
            df_gb[kernel].append(df.values[0, 1])  # only need stddev

    df_gb = pd.DataFrame(df_gb)
    df_gb.to_csv("%s_img_global_uncertainty.csv" % (modellist[0]))

if __name__ == '__main__':
    device = torch.device("cuda:7")
#     # load data set
#     # namelist = ['RBF_Kernel', 'Matern_Kernel', 'Periodic_Kernel', 'Stock50', 'SmartMeter', 'HousePricing']
#     namelist = ['Periodic_Kernel']
#     data, namelist = load_data(namelist)
#     # # define model list
#     # modellist = ['NP', 'ANP', 'GBCoNP']
#     modellist = [
#     # 'NP',
#     'ANP_zdim4',
#         'GBCoNP_zdim4',
#     ]
#     model, modellist, modeldict = define_load_model(data, modellist, sample_size=1)
#     visualize_global_uncertainty(model, data, namelist, modellist, sample_size = 7)

    # global uncertainty - context set size sensitivity analysis
    # df = gb_cntxt_correlation(data, modeldict, device)
#
    # namelist = ['MNIST']
    # namelist = ['MNIST', 'SVHN', 'CelebA32']
    # data, namelist = load_img_data(namelist)
    # modellist = [
#         'NP',
#         'ANP_zdim4',
#         'GBCoNP_zdim4',
#     ]
#     # # # modellist = ['NN','CNP', 'ACNP','ConvCNP','NP', 'ANP','ConvNP']
#     model, modellist, modeldict = define_load_img_model(data, modellist, sample_size=10)
#     for _ in range(1):
#         visualize_img_global_uncertainty(model, data, namelist, modellist)
#     # global uncertainty - context set size sensitivity analysis
#     df = gb_cntxt_img_correlation(data, modeldict, device)
#
    namelist = ['Covid']
    data, namelist = load_covid(namelist)
    modellist = [
                # 'ANP_zdim4',
                'GBCoNP_zdim4'
        ]
    model, modellist = define_load_covid_model(data, modellist, sample_size=1)
#     # plot_covid_by_model(model, data, namelist, modellist)
    for _ in range(1):
        visualize_covid_global_uncertainty(model, data, namelist, modellist, sample_size=5)
#     # gp_demo_plot()
#     # gp_demo_plot()
#     # z_prior_plot()

def plot_correlation():
    import seaborn as sns
    import matplotlib.ticker as ticker
    sns.set_theme(style="dark", color_codes=True)
    import matplotlib.pyplot as plt
    # model_name = 'ANP'
    model_name = 'NP'
    # model_name = 'GBCoNP'
    # df_gb = pd.read_csv("%s_global_uncertainty.csv"%(model_name), index_col=0)
    df_gb = pd.read_csv("%s_img_global_uncertainty.csv"%(model_name), index_col=0)
    fig, axs = plt.subplots()

    res = sns.lineplot(
        data=df_gb,
        markers=True, dashes=False,
        linewidth=3,
        markersize=10,
    )
    # plt.show()
    axs.grid("on", linewidth=2)
    axs.tick_params(labelsize=10)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.tight_layout()
    res.legend(loc='upper right', ncol=2)
    res.set(xlabel="Context set size", ylabel="$\sigma_z$")
    plt.savefig("context_size_%s.png"%model_name)
    plt.show()

# if __name__ == '__main__':
#     plot_correlation()