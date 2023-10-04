import numpy as np
import os
import random
import time


def sensing(A, M):
    b = np.trace(np.einsum('ijk,jh->ihk', A, M), axis1=0, axis2=1)
    return b


def create_data(d, r, n):
    if not os.path.exists(os.path.join('data', 'M_data_' + str(d) + '.npy')):
        us = np.random.normal(loc=0, scale=1/d, size=(d, r))
        M = np.matmul(us, us.T)
        np.save(os.path.join('data', 'M_data_' + str(d) + '.npy'), M)
    else:
        M = np.load(os.path.join('data', 'M_data_' + str(d) + '.npy'))
    if not os.path.exists(os.path.join('data', 'A_data_' + str(d) + '_' + str(n) + '.npy')):
        A = np.random.normal(loc=0, scale=1, size=(d, d, n))
        np.save(os.path.join('data', 'A_data_' + str(d) + '_' + str(n) + '.npy'), A)
    else:
        A = np.load(os.path.join('data', 'A_data_' + str(d) + '_' + str(n) + '.npy'))
    b0 = sensing(A, M)
    return M, A, b0


def initialization(d, r, M, compare):
    if not compare or not os.path.exists(os.path.join('data', 'init_' + str(d) + '.npy')):
        u0 = np.random.normal(loc=0, scale=1, size=(d, 1))
        np.save(os.path.join('data', 'init_' + str(int(time.time())) + '.npy'), u0)
    else:
        u0 = np.load(os.path.join('data', 'init_' + str(d) + '.npy'))
    alpha = 0.99 * np.max(np.absolute(np.linalg.eigvals(M))) / np.sqrt(0.000001 + np.sum(u0 * u0))
    u0 = alpha * u0
    x = np.zeros((d, r))
    x[:, 0] = u0.squeeze()
    return x


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


def get_fval(x, A, b0):
    b = sensing(A, np.matmul(x, x.T))
    fval = 0.5 * np.sum((b - b0) * (b - b0))
    return fval


def get_distance(x, M):
    diff = np.matmul(x, x.T) - M
    Mscale = np.trace(np.matmul(M, M.T))
    dist = np.trace(np.matmul(diff, diff.T)) / Mscale
    return dist


def get_gradient(x, A, b0):
    b = sensing(A, np.matmul(x, x.T))
    gx1 = np.einsum('ijk,jh->ihk', A + np.transpose(A, (1, 0, 2)), x)
    gx = np.mean(gx1 * (b - b0), axis=2)
    return gx


def get_hessian_eig(x, A, b0):
    d, r = x.shape
    n = len(b0)
    b = sensing(A, np.matmul(x, x.T))
    hx1 = np.mean((A + np.transpose(A, (1, 0, 2))) * (b - b0), axis=2)
    block_list = []
    for i in range(r):
        row = []
        for j in range(r):
            if j == i:
                row.append(hx1.copy())
            else:
                row.append(np.zeros((d, d)))
        block_list.append(row)
    hx1 = np.block(block_list)
    gx = np.einsum('ijk,jh->ihk', A + np.transpose(A, (1, 0, 2)), x)
    gx = np.transpose(gx, (1, 0, 2)).reshape((r * d, -1))
    hx2 = np.matmul(gx, gx.T) / n
    hx = hx1 + hx2
    eigval = np.min(np.real(np.linalg.eigvals(hx)))
    return eigval


def consensus(targets, W):
    n, d, r = targets.shape
    ans = np.matmul(W, targets.reshape((n, -1)))
    ans = ans.reshape((n, d, r))
    return ans
