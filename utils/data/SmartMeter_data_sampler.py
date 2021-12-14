from pathlib import Path
import pandas as pd
import collections
import numpy as np
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

def get_smartmeter_df(indir=Path('/share/scratch/xuesongwang/metadata/SmartMeters'), use_logy=False):
    csv_files = indir / 'halfhourly_dataset/block_0.csv'
    df = pd.read_csv(csv_files, parse_dates=[1], na_values=['Null'])
    # print(df.info())

    df = df.groupby('tstp').mean()
    df['tstp'] = df.index
    df.index.name = ''

    # Load weather data
    # df_weather = pd.read_csv(indir / 'weather_hourly_darksky.csv', parse_dates=[3])

    # use_cols = ['visibility', 'windBearing', 'temperature', 'time', 'dewPoint',
    #             'pressure', 'apparentTemperature', 'windSpeed',
    #             'humidity']
    # df_weather = df_weather[use_cols].set_index('time')
    #
    # # Resample to match energy data
    # df_weather = df_weather.resample('30T').ffill()
    #
    # # Normalise
    # weather_norms = dict(mean={'visibility': 11.2,
    #                            'windBearing': 195.7,
    #                            'temperature': 10.5,
    #                            'dewPoint': 6.5,
    #                            'pressure': 1014.1,
    #                            'apparentTemperature': 9.2,
    #                            'windSpeed': 3.9,
    #                            'humidity': 0.8},
    #                      std={'visibility': 3.1,
    #                           'windBearing': 90.6,
    #                           'temperature': 5.8,
    #                           'dewPoint': 5.0,
    #                           'pressure': 11.4,
    #                           'apparentTemperature': 6.9,
    #                           'windSpeed': 2.0,
    #                           'humidity': 0.1})
    #
    # for col in df_weather.columns:
    #     df_weather[col] -= weather_norms['mean'][col]
    #     df_weather[col] /= weather_norms['std'][col]
    #
    # df = pd.concat([df, df_weather], 1).dropna()
    #
    # # Also find bank holidays
    # df_hols = pd.read_csv(indir / 'uk_bank_holidays.csv', parse_dates=[0])
    # holidays = set(df_hols['Bank holidays'].dt.round('D'))
    #
    # df['holiday'] = df.tstp.apply(lambda dt: dt.floor('D') in holidays).astype(int)
    #
    # # Add time features
    # time = df.tstp
    # df["month"] = time.dt.month / 12.0
    # df['day'] = time.dt.day / 310.0
    # df['week'] = time.dt.week / 52.0
    # df['hour'] = time.dt.hour / 24.0
    # df['minute'] = time.dt.minute / 24.0
    # df['dayofweek'] = time.dt.dayofweek / 7.0

    # Drop nan and 0's
    df = df[df['energy(kWh/hh)'] != 0]
    df = df.dropna()

    if use_logy:
        df['energy(kWh/hh)'] = np.log(df['energy(kWh/hh)'] + 1e-4)
    df = df.sort_values('tstp')

    # split data with respect to time
    n_split = -int(len(df) * 0.1)
    df_train = df[:3 * n_split]
    df_val = df[3 * n_split:n_split]
    df_test = df[n_split:]
    print("load SmartMeter success!")
    return df_train, df_val, df_test


class SmartMeter(Dataset):
    def __init__(self, df, max_length = 200, n_samples = 1):
        self.df = df
        self.max_length = max_length
        self.n_samples = int(n_samples)
        self.label_names = ['energy(kWh/hh)']

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
        i = np.random.randint(len(self.df) - self.max_length)
        rows = self.df.iloc[i: i + self.max_length].copy()
        rows['tstp'] = (rows['tstp'] - rows['tstp'].iloc[0]).dt.total_seconds() / 86400.0
        rows = rows.sort_values('tstp')

        # make sure tstp, which is our x axis, is the first value
        # columns = ['tstp'] + list(set(rows.columns) - set(['tstp'])) + ['future']
        # rows['future'] = 0.
        # rows = rows[columns]

        x = rows[['tstp']].copy()
        y = rows[self.label_names].copy()

        xrange = (x.values[0], x.values[-1])

        y = self.normalize(y.values).astype('float32')
        x = self._postprocessing_features(x.values, xrange).astype('float32')
        return x, y

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    df = get_smartmeter_df()
    dataset = SmartMeter(df[0])
    temp = dataset[3]