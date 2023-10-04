import numpy as np
import pandas as pd
import os
import random
import time


def load_data():
    if not os.path.exists('ml-100k.csv'):
        raise NotImplementedError('Dataset Not Exists!')
    data = pd.read_csv('ml-100k.csv', header=None)
    data = data.to_numpy(dtype=float) / 5.0
    return data


def data_distribution(n, nw, method):
    if method == 'random':
        idx = sorted(random.sample(range(1, n), nw - 1))
        return [0] + idx + [n]
    elif method == 'dirichlet':
        distributions = np.random.dirichlet(np.repeat(0.3, nw))
        distributions = distributions + 0.05 # avoid empty dataset
        distributions = distributions / distributions.sum()
        distributions = (np.cumsum(distributions) * n).astype(int)[:-1]
        return [0] + list(distributions) + [n]
    else:
        raise NotImplementedError('Not Implemented Yet!')


def divide_number(n):
    r = int(np.floor(np.sqrt(n)))
    while r > 1:
        if n % r == 0:
            c = n // r
            return r, c
        r = r - 1
    return 1, n


def create_network(n, choice):
    if choice == 'ring':
        W = (1/3) * np.identity(n)
        for i in range(n):
            left = (i + n - 1) % n
            right = (i + 1) % n
            W[i, left], W[i, right] = 1/3, 1/3
    elif choice == 'toroidal':
        row, col = divide_number(n)
        if row <= 2: raise NotImplementedError('Cannot use this topology!')
        W = (1/5) * np.identity(n)
        for i in range(n):
            r, c = divmod(i, col)
            lc, rc = (c + col - 1) % col, (c + 1) % col
            W[i, r * col + lc], W[i, r * col + rc] = 1/5, 1/5
            ur, dr = (r + row - 1) % row, (r + 1) % row
            W[i, ur * col + c], W[i, dr * col + c] = 1/5, 1/5
    elif choice == 'exponential':
        hop = int(np.ceil(np.log2(n / 2)))
        degree = 2 * hop + 1
        W = (1 / degree) * np.identity(n)
        for i in range(n):
            for j in range(hop):
                step = 1 << j
                left = (i - step + n) % n
                right = (i + step) % n
                W[i, left], W[i, right] = (1 / degree), (1 / degree)
    else:
        raise NotImplementedError('Not Implemented Yet!')
    return W


def get_fval(xu, xv, data):
    diff = np.matmul(xu, xv.T) - data
    fval = np.sum(diff * diff)
    return fval / xu.shape[0]


def get_gradient(xu, xv, data):
    diff = np.matmul(xu, xv.T) - data
    gu = 2 * np.matmul(diff, xv)
    gv = 2 * np.matmul(diff.T, xu)
    return gu, gv


def consensus(targets, W):
    n, d, r = targets.shape
    ans = np.matmul(W, targets.reshape((n, -1)))
    ans = ans.reshape((n, d, r))
    return ans
