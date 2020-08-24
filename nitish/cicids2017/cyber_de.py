"""
We are doing something basic to get started.  Given a matrix (cyber data)
where the rows are instances and columns are features we estimate the 'linear' 
dimension.  This is just to get the plumbing going and then we evolve.
"""
import numpy as np
import pandas as pd
import os
import os.path
import sys

# imports for PCA
from sklearn.preprocessing import StandardScaler

# Imports for plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# if a singular value is greater than 1% it is consider signal or else it is 
# noise.  This is just to build up and move on quickly.
# ---------------------------------------------------------------------------
def count_gt_threshold(z, threshold):
    tot = sum(z)
    z_pct = [(i/tot) for i in sorted(z, reverse=True)]
    z_gt_theta = [i for i in z_pct if i >= threshold]
    return len(z_gt_theta)

# -------------------------------------------------
# A routine that returns standardized data from a 
# dataframe 
# -------------------------------------------------
def get_data(df):
    features = df.columns
    # ignore the first column (timesamp) and last column (label)
    features = features[1:len(features)-1]
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    return x


def get_singular_values(x):
    U,S,V = np.linalg.svd(x)
    return S

# ----------------------------------------------------------------------------------
# Instead of listdir, we are hardcoding it to see if there is temporal relationship
# between the variables
# ----------------------------------------------------------------------------------
def get_input_files():
    files = ['all_data.csv']
    #files = ['pre_attack_1.csv', 'pre_attack_2.csv', 'pre_attack_3.csv', 
    #         'attack_1.csv', 'attack_2.csv', 'attack_3.csv', 
    #         'post_attack_1.csv', 'post_attack_2.csv', 'post_attack_3.csv',
    #         'steady_state.csv']
    return files

def plot_dimensions_over_time(times, dimensions, freq):
    fig, ax = plt.subplots()
    ax.plot(times, dimensions)
    plt.title(f'Number of Dimensions in {freq} Time Intervals')
    plt.xlabel('Time')
    plt.ylabel('Number of Dimensions')
    ax.xaxis_date()
    ax.set_xticks(times)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.axvspan(*mdates.datestr2num(['2017-06-07 9:15:00', '2017-06-07 10:00:00']), color='gray', alpha=0.4)
    ax.axvspan(*mdates.datestr2num(['2017-06-07 10:15:00', '2017-06-07 10:35:00']), color='gray', alpha=0.4)
    ax.axvspan(*mdates.datestr2num(['2017-06-07 10:40:00', '2017-06-07 10:41:00']), color='gray', alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(freq + '_dimensions_over_time.png')
    plt.close()

# the program main
if __name__ == '__main__':
    basedir = r"./semantic_encoding"
    threshold = 0.01
    files = get_input_files()

    date_range_freqs = ['1min', '5min', '10min', '15min']

    # Check that files exist first
    for f in files:
        if not os.path.exists(f'{basedir}/{f}'):
            print(f'{f} not found.  Please run gunzip {basedir}/{f}.gz')
            sys.exit()

    for file in files:
        df = pd.read_csv(os.path.join(basedir, file))

        ###
        # Create time index and use it to obtain
        # min/max datetime
        ###
        df.timestamp = pd.to_datetime(df.timestamp)
        df = df.set_index(df.timestamp)
        df = df.sort_index()
        min_date = df.index.min()
        max_date = df.index.max() + pd.Timedelta(minutes=1)

        for freq in date_range_freqs:
            date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            interval_start = date_range[0]
            dimensions = list()
            times = list()

            for interval_end in date_range[1:]:
                current_df = df.loc[interval_start:interval_end]
                if current_df.empty:
                    print(f'No data for range {interval_start}:{interval_end}')
                    interval_start = interval_end
                    continue

                x = get_data(current_df)
                S = get_singular_values(x)
                gte_dim = count_gt_threshold(S, threshold)
                dimensions.append(gte_dim)
                times.append(interval_start)

                print(f"Interval: {interval_start}:{interval_end} linear dimension is {gte_dim}")
                interval_start = interval_end

            plot_dimensions_over_time(times, dimensions, freq)
