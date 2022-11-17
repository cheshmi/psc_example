import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit



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


def plot():
    df = pd.read_csv("/Users/kazem/development/psc_example/psc.csv")
    time_scalarized = df["Scalarized"].values
    time_blas = df["BLAS"].values
    time_pscg = df["PSC G"].values
    time_psc2d = df["PSC 2D"].values
    time_psc2d4 = df["PSC 2D-4"].values
    nnz = df["NNZ"].values

    #plt.scatter(nnz, time_scalarized)

    plt.scatter(nnz, time_blas, marker='^', c='black')
    #plt.scatter(nnz, time_psc2d, marker='o', c='m')
    plt.scatter(nnz, time_psc2d4, marker='o', c='red')



    plt.xlabel("Number of NonZero Elements")
    plt.ylabel("Time (sec)")
    plt.title("Vectorization using PSC vs. BLAS in single-thread")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
