# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, jit


@njit
def fsign(f):
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

@jit
def lasso_nb(X, y, alpha, maxiter=200):
    n, p = X.shape
    beta = np.zeros(p)
    R = y.copy()
    norm_cols_X = (X ** 2).sum(axis=0)
    for n_iter in range(maxiter):
        for ii in range(p):
            beta_ii = beta[ii]
            # Get current residual
            if beta_ii != 0.:
                R += X[:, ii] * beta_ii
            tmp = np.dot(X[:, ii], R)
            # Soft thresholding
            beta[ii] = fsign(tmp) * max(abs(tmp) - alpha, 0) / norm_cols_X[ii]

            if beta[ii] != 0.:
                R -= X[:, ii] * beta[ii]
    return beta