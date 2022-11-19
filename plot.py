import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys

font = {'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


def plot(csv_name):
    df = pd.read_csv(csv_name)
    time_scalarized = df["Scalarized"].values
    time_blas = df["BLAS"].values*1e3
    time_pscg = df["PSC G"].values*1e3
    time_psc2d = df["PSC 2D"].values*1e3
    time_psc2d4 = df["PSC 2D-4"].values*1e3
    nnz = df["NNZ"].values
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,12))

    ax0.scatter(nnz, time_blas, marker='^', c='black', label="BLAS-based SpMV")
    #ax0.scatter(nnz, time_psc2d, marker='o', c='m')
    ax0.scatter(nnz, time_psc2d4, marker='o', c='red', label="PSC-based SpMV")

    #ax1.scatter(nnz, time_scalarized / time_blas, c='blue')
    ax1.scatter(nnz, time_blas / time_psc2d4, c='m')

    print("Average speedup is: ", np.average(time_blas/time_psc2d4))

    ax1.plot(nnz, np.ones(len(nnz)), c='black')
    ax1.set_yticks(list(range(1, 6, 2)) + list(range(10,50, 5)))
    ax0.set(xlabel="Number of NonZero Elements", ylabel="Time (milli-seconds)")
    ax1.set(xlabel="Number of NonZero Elements", ylabel="PSC-based SpMV speedup over BLAS-based SpMV")
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax0.legend(loc='upper left')
    fig.suptitle('Vectorization using PSC vs. BLAS in single-thread', fontsize=30)
    plt.show()
    plt.savefig("psc_blas.png")


if __name__ == '__main__':
    csv_name = sys.argv[1] # the path to the CSV file
    plot(csv_name)
