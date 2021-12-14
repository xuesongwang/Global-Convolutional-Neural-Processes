#Data Analyses Libraries
import pandas as pd
from urllib.request import urlopen
import json

import plotly.express as px

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from utils.data.imgs import CovidMap

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os


def save_log_relative_growth():
    root = "/share/scratch/xuesongwang/metadata/Covid/"
    usa_cases = pd.read_csv(os.path.join(root, "us-counties.csv"))
    new_df = pd.DataFrame()
    for _, df_county in usa_cases.groupby("fips"):
        df_county
        for back_shift_day in range(0, 15):
            # calculate last 14 days
            df_county_day = df_county['cases'].shift(periods=back_shift_day)
            df_county_day.fillna(0, inplace=True)
            df_county['back'+str(back_shift_day)+'days'] = df_county_day
            # calculate next 14 days
            if back_shift_day > 0:
                df_county_day = df_county['cases'].shift(periods=-back_shift_day)
                df_county_day.fillna(0, inplace=True)
                df_county['next' + str(back_shift_day) + 'days'] = df_county_day

        # find min value
        feature_list = ['back'+str(back_shift_day)+'days' for back_shift_day in range(15)] + \
                       ['next' + str(back_shift_day) + 'days' for back_shift_day in range(1, 15)]
        df_min = df_county[feature_list].min(axis=1)

        # logged relative growth rate
        logged = np.log(df_county[feature_list].values - np.expand_dims(df_min.values, axis=1) + 1)
        log_feature_list = ['Log_rel_'+feature for feature in feature_list]
        df_logged = pd.DataFrame(logged, index=df_county.index, columns=log_feature_list)
        df_county = pd.concat([df_county, df_logged], axis=1)

        # drop first and the last 14 rows, as they have 0 as min values for those days, and the log-relative growth can go row
        index = df_county.index
        df_county.drop(index[:15], inplace=True)
        df_county.drop(index[-15:], inplace=True)

        # form new df
        new_df = pd.concat([new_df, df_county], axis=0, ignore_index=True)

    # new_df.to_csv(os.path.join(root, "usa_new_cases.csv"))


def save_images(root = "/share/scratch/xuesongwang/metadata/Covid/"):
    usa_cases = pd.read_csv(os.path.join(root, "usa_new_cases.csv"))
    usa_cases['fips'] = usa_cases['fips'].astype(str).str.rjust(5, '0')
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    for date, county in tqdm(usa_cases.groupby('date')):
        count = county.back7days.sum()
        print("date: %s, total counties: %s, new cases in the last 7 days: %s" % (date, county.shape[0], count))
        # context images
        for shift_days in [7, 3, 0]:
            fig = px.choropleth(county, geojson=counties, locations='fips', color='Log_rel_back%ddays' % shift_days,
                                color_continuous_scale="gray",
                                range_color=(0, 13),
                                scope="usa")
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              coloraxis_colorbar=dict(title="log"))
            fig.update_traces(marker_line_width=0)
            # py.offline.iplot(fig)
            fig.write_image("covid_usa_back%d/%s.png" % (shift_days, date))
        # target images
        fig = px.choropleth(county, geojson=counties, locations='fips', color='Log_rel_next7days',
                            color_continuous_scale="gray",
                            range_color=(0, 13),
                            scope="usa")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          coloraxis_colorbar=dict(title="log"))
        fig.update_traces(marker_line_width=0)
        # py.offline.iplot(fig)
        fig.write_image("covid_usa_next7/%s.png" % (date))


def plot_sample_spatial(sample):
    """
    sample: shape (*, C, T, W, H)
    """
    for t in range(sample.shape[2]):
        img_grid = make_grid(sample[:,:,t,:,:], nrow=13, pad_value=1,)
        img = img_grid.permute(1, 2, 0).numpy()
        ax = plt.imshow(img[:,:,0], cmap=plt.get_cmap('Reds'))
        plt.show()

def plot_sample_temporal(sample):
    """
        sample: shape (*, C, T, W, H)
    """
    n_samples = 20
    values = np.zeros((n_samples, sample.shape[2]))
    for n in range(n_samples):
        patch = np.random.randint(sample.shape[0])
        width = np.random.randint(sample.shape[-2])
        height = np.random.randint(sample.shape[-1])
        values[n] = sample[patch,0,:,width, height]
    step = np.array([-7, -3, 0, 7])
    plt.plot(step, values.T)
    scatter_x = np.repeat(step[None,:], n_samples, axis=0).reshape(-1)
    scatter_y = values.reshape(-1)
    plt.scatter(scatter_x, scatter_y)
    plt.show()



if __name__ == '__main__':
    # save_images()
    # print("finished")
    root = '/home/xuesongwang/PycharmProject/NPF/utils/data/covid_usa_back7'
    covid_dataset = CovidMap(split="train")
    sample = covid_dataset[np.random.randint(len(covid_dataset))]
    plot_sample_spatial(sample)
    plot_sample_temporal(sample)
