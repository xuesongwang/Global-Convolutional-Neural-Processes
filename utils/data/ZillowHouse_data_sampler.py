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

def get_housepricing_df(indir=Path('/share/scratch/xuesongwang/metadata/HousePricing')):
    file_csv = indir / 'City_Zhvi_AllHomes.csv'
    df = pd.read_csv(file_csv)
    df =df.dropna() # remove month without temperature values

    # print(df.info())
    np.random.seed(0)
    city_shuffle = np.random.permutation(len(df))
    df = df.iloc[city_shuffle]

    test_count = int(len(df)*0.1)
    df_train = df[:-2*test_count]
    df_val = df[-2*test_count:-test_count]
    df_test = df[-test_count:]
    print("load HousePricing success!")
    return df_train, df_val, df_test



class HousePricing(Dataset):
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
        df_city = self.df.iloc[i, 9:].astype('float')
        random_date_index = np.random.randint(df_city.shape[0] - self.max_length)

        x = np.arange(-1, 1, 2/self.max_length)  # index with yearly unit
        y = df_city.iloc[random_date_index: (random_date_index + self.max_length)]
        # xrange = (x.values[0], x.values[-1])

        y = np.expand_dims(self.normalize(y.values), axis=-1).astype('float32')
        x = np.expand_dims(x, axis=-1).astype('float32')
        return x, y

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    df = get_housepricing_df()
    dataset = HousePricing(df[0])
    sample = dataset[3]