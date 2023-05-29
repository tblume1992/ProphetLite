# -*- coding: utf-8 -*-
"""
Based On Nixtla code I think
"""
from numba import njit
import numpy as np


@njit
def get_fourier_series(length, seasonal_period, fourier_order):
    x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
    t = np.arange(1, length + 1).reshape(-1, 1)
    x = x * t
    fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return fourier_series

@njit
def get_future_fourier_series(forecast_horizon, length, seasonal_period, fourier_order):
    x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
    t = np.arange(1, forecast_horizon + length + 1).reshape(-1, 1)
    x = x * t
    fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return fourier_series[-forecast_horizon:, :]


#%%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seasonality = ((np.cos(np.arange(1, 101))*10 + 50))
    np.random.seed(100)
    true = np.linspace(-1, 1, 100)
    noise = np.random.normal(0, 1, 100)
    y = true + noise + seasonality
    plt.plot(y)
    basis = get_fourier_series(len(y), 25, 5)
    future_basis = get_future_fourier_series(24, len(y), 25, 5)
    full = np.append(basis, future_basis, axis=0)
    plt.plot(full + y[0])