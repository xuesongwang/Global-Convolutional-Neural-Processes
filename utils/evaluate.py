import torch
from skorch.dataset import unpack_data, uses_placeholder_y
import numpy as np
from .helpers import set_seed
from tqdm import tqdm
__all__ = ["eval_loglike", "test_with_loglikelihood", "testing",
           "compute_loss", "compute_MSE"]


def eval_loglike(trainer, dataset, seed=123, **kwargs):
    """Return the log likelihood for each image in order."""
    set_seed(seed)  # make sure same order and indices for cntxt and trgt
    trainer.module_.to(trainer.device)
    old_reduction = trainer.criterion_.reduction
    trainer.criterion_.reduction = None
    y_valid_is_ph = uses_placeholder_y(dataset)
    all_losses = []

    trainer.notify("on_epoch_begin", dataset_valid=dataset)
    for data in tqdm(trainer.get_iterator(dataset, training=False)):
        Xi, yi = unpack_data(data)
        yi_res = yi if not y_valid_is_ph else None
        trainer.notify("on_batch_begin", X=Xi, y=yi_res, training=False)
        step = trainer.validation_step(Xi, yi, **kwargs)
        trainer.notify("on_batch_end", X=Xi, y=yi_res, training=False, **step)
        all_losses.append(-step["loss"])  # use log likelihood instead of NLLL

    trainer.criterion_.reduction = old_reduction
    return torch.cat(all_losses, dim=0).detach().cpu().numpy()


def compute_loss(mean, var, y_target, issum=True):
    from torch.distributions import Normal
    dist = Normal(loc=mean, scale=var)

    if mean.shape[0] != y_target.shape[0]:  # z_sample
        y_target = y_target.unsqueeze(0).expand(mean.size())
    log_prob = dist.log_prob(y_target)
    loss = - torch.mean(torch.sum(log_prob, dim=-2)) if issum \
        else - torch.mean(log_prob)
    return loss

def compute_MSE(mean, y_target):
    import torch.nn as nn
    criterion = nn.MSELoss()
    if mean.shape[0] != y_target.shape[0]: # z_sample
        y_target = y_target.unsqueeze(0).expand(mean.size())
    mean_mse = criterion(mean, y_target)
    return mean_mse

def testing(data_test, trainer, evaluate_z=False):
    from tqdm import tqdm
    total_ll = []
    total_mse = []
    total_zmean = []
    total_zstd = []
    device = trainer.device
    model = trainer.module_.eval().to(device)
    with torch.no_grad():
        for data in tqdm(trainer.get_iterator(data_test, training=False)):
            x_context = data[0]['X_cntxt'].to(device)
            y_context = data[0]['Y_cntxt'].to(device)
            x_target = data[0]['X_trgt'].to(device)
            y_target = data[0]['Y_trgt'].to(device)
            dist, _, zprior, _ = model(x_context, y_context,
                                  x_target)
            mean = dist.mean
            var = dist.stddev
            total_ll.append(-compute_loss(mean, var, y_target, issum=False).item())
            total_mse.append(compute_MSE(mean,y_target).item())
            if evaluate_z:
                z_mean = zprior.mean
                z_std = zprior.stddev
                total_zmean.append(torch.mean(z_mean).item())
                total_zstd.append(torch.mean(z_std).item())
    return np.mean(total_ll), np.mean(total_mse), np.mean(total_zmean), np.mean(total_zstd)




def test_with_loglikelihood(dataset, trainer):
    import pandas as pd
    total_loss = []
    total_mse = []
    for _ in range(6):
        test_loss, test_mse, _, _ = testing(dataset, trainer)
        total_loss.append(test_loss)
        total_mse.append(test_mse)
    df = pd.DataFrame()
    df['loglikelihood'] = total_loss
    df['MSE'] = total_mse
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_loss), np.std(total_loss)))
    # print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    return df


def quantify_global_uncertainty(dataset, trainer):
    import pandas as pd
    _, _, zmean, zstd = testing(dataset, trainer, evaluate_z=True)

    df = pd.DataFrame()
    df['zmean'] = [zmean]
    df['zstd'] = [zstd]
    print("for 6 runs, mean: %.4f, std:%.4f" % (zmean, zstd))
    return df