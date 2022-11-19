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
    df = pd.read_csv("//home/kazem/development/psc_example/psc.csv")
    time_scalarized = df["Scalarized"].values
    time_blas = df["BLAS"].values
    time_pscg = df["PSC G"].values
    time_psc2d = df["PSC 2D"].values
    time_psc2d4 = df["PSC 2D-4"].values
    nnz = df["NNZ"].values
    fig, (ax0, ax1) = plt.subplots(1, 2)
    #ax = plt.gca()
  #  plt.scatter(nnz, time_scalarized, c='y')

    ax0.scatter(nnz, time_blas, marker='^', c='black')
    #ax.scatter(nnz, time_psc2d, marker='o', c='m')
    ax0.scatter(nnz, time_psc2d4, marker='o', c='red')

    #ax1.scatter(nnz, time_scalarized / time_blas, c='blue')
    ax1.scatter(nnz, time_blas / time_psc2d4, c='red')

    print( np.average(time_blas/ time_psc2d4) )

    #axs[0, 0].set_yscale('log')

    ax0.set(xlabel="Number of NonZero Elements", ylabel="Time (sec)")
    plt.title("Vectorization using PSC vs. BLAS in single-thread")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
