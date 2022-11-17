
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import numpy as np
import random
from scipy import sparse
import scipy.io as sio


def a1(n_blk, i0_l, mx_b_s, fname):
    num_block = int(n_blk)
    i0_len, i1_len = int(i0_l), 0
    max_blk_size, max_interval_size = int(mx_b_s), 50
    blk_size_array, interval_size_array, loc_array = [], [], [0]
    for i in range(num_block):
        blk_size_array.append(random.randint(1, max_blk_size))
        interval_size_array.append(random.randint(1, max_interval_size))
        loc_array.append(loc_array[i] + blk_size_array[i] + interval_size_array[i])
        i1_len += (blk_size_array[i] + interval_size_array[i])
    dense_mtx = np.zeros((i0_len, i1_len))
    for i in range(num_block):
        for j in range(loc_array[i], loc_array[i]+blk_size_array[i], 1):
            for k in range(i0_len):
                dense_mtx[k][j] = random.randint(1, 10)

    csr_mtx = sparse.csr_matrix(dense_mtx)
    csc_mtx = sparse.csc_matrix(dense_mtx)
    sio.mmwrite(fname+"mat_"+str(num_block)+"_"+str(i0_len)+"_"+str(max_blk_size), csc_mtx)


if __name__ == '__main__':
    fname, nb, i0, mx_bs = '', 10, 4, 20
    if len(sys.argv) > 4:
        nb = sys.argv[1]
        i0 = sys.argv[2]
        mx_bs = sys.argv[3]
        fname = sys.argv[4]
    a1(nb, i0, mx_bs, fname)
