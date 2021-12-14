import numpy as np
import collections
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
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

def get_NIFTY_df(directory = '/share/scratch/xuesongwang/metadata/Stock Market Data'):
    filepathlist = os.listdir(directory)
    train_df = []
    val_df = []
    test_df = []
    for fileindex, file in enumerate(filepathlist):
        # print("file name:%s"%file)
        if file in ['NIFTY50_all.csv', 'stock_metadata.csv',]:
            continue
        df = pd.read_csv(os.path.join(directory,file))
        df.set_index("Date", drop=False, inplace=True)
        df = df['VWAP']
        # mean = df[:'2016-11-28'].mean()
        # std = df[:'2016-11-28'].std()
        # train_df.append(normalize(df[:'2016-11-28'], mean, std))
        # val_df.append(normalize(df['2016-11-28': '2017-11-29'], mean, std))
        # test_df.append(normalize(df['2017-11-29':], mean, std))
        if fileindex < 37: # the first 37 stocks
            train_df.append(df)
        elif fileindex > 47:
            test_df.append(df)
        else:
            val_df.append(df)
    print("load NIFTY50 success!")
    return train_df, val_df, test_df

class NIFTY(Dataset):
    def __init__(self, df, max_length = 200, n_samples = 1):
        self.df = df
        self.max_length = max_length
        self.n_samples = int(n_samples)

    def get_rows(self, i):
        stock = self.df[i%len(self.df)]
        try:
            index = np.random.randint(stock.shape[0] - self.max_length)
        except:
            print("stock length:",stock.shape[0])
        slice = stock.iloc[index: index + self.max_length].copy()
        x = (pd.to_datetime(slice.index) -  pd.to_datetime(slice.index[0])).days/30 # derive day difference, then normalize by month
        return x, slice, (x.values[0], x.values[-1])

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
        stock = self.df[i % len(self.df)]
        try:
            index = np.random.randint(stock.shape[0] - self.max_length)
        except:
            print("stock length:", stock.shape[0])
        slice = stock.iloc[index: index + self.max_length].copy()
        x = (pd.to_datetime(slice.index) - pd.to_datetime(
            slice.index[0])).days / 30  # derive day difference, then normalize by month
        y = slice
        xrange = (x.values[0], x.values[-1])

        y = np.expand_dims(self.normalize(y.values), axis=-1).astype('float32')
        x = np.expand_dims(self._postprocessing_features(x.values, xrange), axis=-1).astype('float32')
        return x, y

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    df = get_NIFTY_df()
    dataset = NIFTY(df[0])
    sample = dataset[3]