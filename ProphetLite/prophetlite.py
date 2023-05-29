# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from ProphetLite.model import lasso_nb
from ProphetLite.fourier import get_fourier_series, get_future_fourier_series
from ProphetLite.linear_basis import get_basis, get_future_basis


class ProphetLite:

    def __init__(self,):
        self.predict_step = False

    def scale(self, y):
        self.y_std = np.std(y)
        self.y_mean = np.mean(y)
        return (y - self.y_mean) / self.y_std

    def inverse_scale(self, y):
        return y * self.y_std + self.y_mean

    def fit(self,
            y,
            seasonal_period=None,
            n_changepoints=25,
            fourier_order=10,
            alpha=.01,
            max_iter=200,
            decay=-1):
        self.y = y
        y = self.scale(y)
        self.training_length = len(y)
        self.n_changepoints = n_changepoints
        self.fourier_order = fourier_order
        n = len(y)
        if n < n_changepoints:
            n_changepoints = n - 1
        linear_basis, self.slopes, self.diff = get_basis(y=y, n_changepoints=n_changepoints, decay=decay)
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
            fourier_series = []
            for i in seasonal_period:
                 fourier_series.append(get_fourier_series(n,
                                                          i,
                                                          fourier_order))
            fourier_series = np.hstack(fourier_series)
            idx = np.argwhere(np.all(fourier_series == 1, axis=0))
            fourier_series = np.delete(fourier_series, idx, axis=1)
            X = np.concatenate((fourier_series, linear_basis), axis=1)
        else:
            X = linear_basis
        self.seasonal_period = seasonal_period
        self.weights = lasso_nb(X, y, alpha, maxiter=max_iter)
        fitted = np.dot(X, self.weights)
        trend_X = X.copy()
        if self.seasonal_period is not None:
            seasonal = X.copy()
            seasonal[:, fourier_series.shape[1]:] = 0
            seasonality = np.dot(seasonal, self.weights)
            trend_X[:, :fourier_series.shape[1]] = 0
        else:
            seasonality = np.zeros(len(y))
        trend = np.dot(trend_X, self.weights)
        seasonality = self.inverse_scale(seasonality)
        trend = self.inverse_scale(trend)
        fitted = self.inverse_scale(fitted)
        resid = fitted - self.y
        self.trend_penalty = max(0, 1 - np.std(resid) / np.std(resid + trend))
        self.fitted = {'seasonality': seasonality,
                       'trend': trend,
                       'yhat': fitted}
        return self.fitted

    def predict(self, forecast_horizon, trend_penalty=False):
        future_linear = get_future_basis(forecast_horizon, self.slopes, self.diff, self.n_changepoints, self.training_length)
        if self.seasonal_period is not None:
            fourier_series = []
            for i in self.seasonal_period:
                 fourier_series.append(get_future_fourier_series(forecast_horizon,
                                                                 self.training_length,
                                                                 i,
                                                                 self.fourier_order))
            fourier_series = np.hstack(fourier_series)
            idx = np.argwhere(np.all(fourier_series == 1, axis=0))
            fourier_series = np.delete(fourier_series, idx, axis=1)
            X = np.concatenate((fourier_series, future_linear), axis=1)
        else:
            X = future_linear
        predicted = np.dot(X, self.weights)
        trend_X = X.copy()
        if self.seasonal_period is not None:
            seasonal = X.copy()
            seasonal[:, fourier_series.shape[1]:] = 0
            seasonality = np.dot(seasonal, self.weights)
            trend_X[:, :fourier_series.shape[1]] = 0
            seas = self.inverse_scale(seasonality)
        else:
            seas = np.zeros(forecast_horizon)
        trend = np.dot(trend_X, self.weights)
        trend = self.inverse_scale(trend)
        predicted = self.inverse_scale(predicted)
        if trend_penalty:
            predicted -= trend
            trend_diff = trend[-1] - trend[0]
            trend = np.linspace(trend[0],
                                      trend[0] + trend_diff* self.trend_penalty,
                                     len(trend))
            predicted += trend
        self.forecast = {'seasonality': seas, 'trend': trend, 'yhat': predicted}
        self.predict_step = True
        return self.forecast

    def plot(self):
        plt.plot(self.y)
        if self.predict_step:
            plt.plot(np.append(self.fitted['yhat'], self.forecast['yhat']))
        else:
            plt.plot(self.fitted['yhat'])
        plt.show()

    def plot_components(self, figsize=(6,4)):
        # plt.plot(self.y)
        if self.predict_step:
            if self.seasonal_period is not None:
                fig, ax = plt.subplots(3, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Seasonality')
                ax[2].set_title('Residuals')
                ax[1].plot(np.append(self.fitted['seasonality'],
                                     self.forecast['seasonality']))
                ax[0].plot(np.append(self.fitted['trend'],
                                     self.forecast['trend']))
                ax[2].plot(self.y - self.fitted['yhat'])
            else:
                fig, ax = plt.subplots(2, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Residuals')
                ax[0].plot(np.append(self.fitted['trend'],
                                     self.forecast['trend']))
                ax[1].plot(self.y - self.fitted['yhat'])
        else:
            if self.seasonal_period is not None:
                fig, ax = plt.subplots(3, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Seasonality')
                ax[2].set_title('Residuals')
                ax[1].plot(self.fitted['seasonality'])
                ax[0].plot(self.fitted['trend'])
                ax[2].plot(self.y - self.fitted['yhat'])
            else:
                fig, ax = plt.subplots(2, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Residuals')
                ax[0].plot(self.fitted['trend'])
                ax[1].plot(self.y - self.fitted['yhat'])
        plt.tight_layout()
        plt.show()
