import numpy as np
import collections
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

# The NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "y_target", "num_total_points", "num_context_points"))

def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min

def get_exoplanet_df(indir=Path('/share/scratch/xuesongwang/metadata/ExoPlanet'), use_logy=False):
    train_csv = indir / 'exoTrain.csv'
    test_csv = indir / 'exoTest.csv'
    df = pd.read_csv(train_csv, na_values=['Null'])
    df =df.drop(labels=['LABEL'], axis=1)
    # print(df.info())

    # split data with respect to time
    n_split = -int(len(df) * 0.1)
    df_train = df[: n_split]
    df_val = df[n_split:]

    df_test = pd.read_csv(test_csv, na_values=['Null'])
    df_test = df_test.drop(labels=['LABEL'], axis=1)
    print("load Exoplanet success!")
    return df_train, df_val, df_test


class Exoplanet(Dataset):
    def __init__(self, df, max_length = 200, n_samples = 1):
        self.df = df
        self.max_length = max_length
        self.n_samples = len(df)

    def _postprocessing_features(self, X, min_max):
        """Convert the features to a tensor, rescale them to [-1,1] and expand."""
        X = rescale_range(X, min_max, (-1, 1))
        return X


    def normalize(self, y):  # normalize on stock-scale
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        y_norm = (y - mean) / std
        return y_norm

    def __getitem__(self, i):
        planet = self.df.iloc[i]
        try:
            index = np.random.randint(planet.shape[0] - self.max_length)
        except:
            print("planet length:", planet.shape[0])
        y = planet.iloc[index: index + self.max_length].copy().values
        x = np.expand_dims(np.arange(-1, 1, 2 / y.size), axis=1).astype('float32')
        y = np.expand_dims(self.normalize(y), axis=-1).astype('float32')
        return x, y

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    dataset = NIFTYReader()
    temp = dataset._get_cache_dfs()
    train_loader = dataset.train_dataloader()
    for i, batch in enumerate(train_loader):
        batch
        print("pause")