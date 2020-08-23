"""
We are doing something basic to get started.  Given a matrix (cyber data)
where the rows are instances and columns are features we estimate the 'linear' 
dimension.  This is just to get the plumbing going and then we evolve.
"""
import numpy as np
import pandas as pd
import os

# imports for PCA
from sklearn.preprocessing import StandardScaler

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
    # ignore the last column, which contains label
    features = features[0:len(features)-1]
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
    files = ['pre_attack_1.csv', 'pre_attack_2.csv', 'pre_attack_3.csv', 
             'attack_1.csv', 'attack_2.csv', 'attack_3.csv', 
             'post_attack_1.csv', 'post_attack_2.csv', 'post_attack_3.csv',
             'steady_state.csv']
    return files

# the program main
if __name__ == '__main__':
    basedir = r"./semantic_encoding"
    threshold = 0.01
    # files = os.listdir(basedir)
    files = get_input_files()
    for file in files:        
        df = pd.read_csv(os.path.join(basedir, file))
        x = get_data(df)
        S = get_singular_values(x)
        gte_dim = count_gt_threshold(S, threshold)
        print(f"File: {file}  linear dimension is {gte_dim}")
