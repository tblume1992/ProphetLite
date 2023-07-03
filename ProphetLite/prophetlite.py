# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from ProphetLite.model import lasso_nb
from ProphetLite.fourier import get_fourier_series, get_future_fourier_series
from ProphetLite.linear_basis import get_basis, get_future_basis
import warnings
warnings.filterwarnings('ignore')


class ProphetLite:

    def __init__(self,):
        self.predict_step = False
        self.trend_size = 0
        self.seasonal_size = 0
        self.exogenous_size = 0


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
            decay=-1,
            exogenous=None):
        self.y = y
        y = self.scale(y)
        self.training_length = len(y)
        self.n_changepoints = n_changepoints
        self.fourier_order = fourier_order
        n = len(y)
        if n < n_changepoints:
            n_changepoints = int(.8 * n - 1)
        linear_basis, self.slopes, self.diff = get_basis(y=y, n_changepoints=n_changepoints, decay=decay)
        self.trend_size = np.shape(linear_basis)[1]
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
            X = np.concatenate((linear_basis, fourier_series), axis=1)
            self.seasonal_size = np.shape(fourier_series)[1]
        else:
            X = linear_basis
        self.seasonal_period = seasonal_period
        if exogenous is not None:
            X = np.concatenate((X, exogenous), axis=1)
            self.exogenous_size = np.shape(exogenous)[1]
        self.weights = lasso_nb(X, y, alpha, maxiter=max_iter)
        fitted = np.dot(X, self.weights)
        trend_X = X.copy()
        if self.seasonal_period is not None:
            seasonal = X.copy()
            seasonal[:, :self.trend_size] = 0
            seasonal[:, self.seasonal_size + self.trend_size:] = 0
            seasonality = np.dot(seasonal, self.weights)
            trend_X[:, self.trend_size:] = 0
            seasonality = self.inverse_scale(seasonality) - self.y_mean
        else:
            seasonality = np.zeros(len(y))

        if exogenous is not None:
            exo_X = X.copy()
            exo_X[:, :self.trend_size + self.seasonal_size] = 0
            exo = np.dot(exo_X, self.weights)
            exo = self.inverse_scale(exo) - self.y_mean
        else:
            exo = np.zeros(len(y))
        trend = np.dot(trend_X, self.weights)
        trend = self.inverse_scale(trend)
        fitted = self.inverse_scale(fitted)
        resid = fitted - self.y
        self.trend_penalty = max(0, 1 - np.std(resid) / np.std(resid + trend))
        self.fitted = {'seasonality': seasonality,
                       'trend': trend,
                       'exogenous': exo,
                       'yhat': fitted}
        return self.fitted

    def predict(self, forecast_horizon, exogenous=None, trend_penalty=False):
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
            X = np.concatenate((future_linear, fourier_series), axis=1)
        else:
            X = future_linear
        if exogenous is not None:
            X = np.concatenate((X, exogenous), axis=1)
        predicted = np.dot(X, self.weights)
        trend_X = X.copy()
        if self.seasonal_period is not None:
            seasonal = X.copy()
            seasonal[:, self.trend_size + self.seasonal_size:] = 0
            seasonal[:, :self.trend_size] = 0
            seasonality = np.dot(seasonal, self.weights)
            trend_X[:, self.trend_size:] = 0
            seas = self.inverse_scale(seasonality) - self.y_mean
        else:
            seas = np.zeros(forecast_horizon)
        if exogenous is not None:
            exo_X = X.copy()
            exo_X[:, :self.trend_size + self.seasonal_size] = 0
            exo = np.dot(exo_X, self.weights)
            exo = self.inverse_scale(exo) - self.y_mean
        else:
            exo = np.zeros(len(y))
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
        self.forecast = {'seasonality': seas, 'trend': trend, 'exogenous': exo, 'yhat': predicted}
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
        plt_num = 2
        if self.predict_step:
            if self.seasonal_period is not None:
                plt_num += 1
                if self.exogenous_size:
                    plt_num += 1
                fig, ax = plt.subplots(plt_num, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Seasonality')
                if self.exogenous_size:
                    plt_num += 1
                    ax[2].set_title('Exogenous')
                    ax[2].plot(np.append(self.fitted['exogenous'],
                                         self.forecast['exogenous']))
                ax[-1].set_title('Residuals')
                ax[1].plot(np.append(self.fitted['seasonality'],
                                     self.forecast['seasonality']))
                ax[0].plot(np.append(self.fitted['trend'],
                                     self.forecast['trend']))
                ax[-1].plot(self.y - self.fitted['yhat'])
            else:
                fig, ax = plt.subplots(plt_num, figsize=figsize)
                ax[0].set_title('Trend')
                ax[-1].set_title('Residuals')
                ax[0].plot(np.append(self.fitted['trend'],
                                     self.forecast['trend']))
                ax[-1].plot(self.y - self.fitted['yhat'])
        else:
            if self.seasonal_period is not None:
                plt_num += 1
                if self.exogenous_size:
                    plt_num += 1
                fig, ax = plt.subplots(plt_num, figsize=figsize)
                ax[0].set_title('Trend')
                ax[1].set_title('Seasonality')
                if self.exogenous_size:
                    plt_num += 1
                    ax[2].set_title('Exogenous')
                    ax[2].plot(self.fitted['exogenous'])
                ax[-1].set_title('Residuals')
                ax[1].plot(self.fitted['seasonality'])
                ax[0].plot(self.fitted['trend'])
                ax[-1].plot(self.y - self.fitted['yhat'])
            else:
                fig, ax = plt.subplots(plt_num, figsize=figsize)
                ax[0].set_title('Trend')
                ax[-1].set_title('Residuals')
                ax[0].plot(self.fitted['trend'])
                ax[-1].plot(self.y - self.fitted['yhat'])
        plt.tight_layout()
        plt.show()

#%%
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from prophet import Prophet
    
    
    df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
    m = Prophet(weekly_seasonality=False)
    m.fit(df, )
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()

    y = df['y'].values



    pl = ProphetLite()
    fitted = pl.fit(y, [365.25]) #To pass multiple seasonalities just add more nu
    predicted = pl.predict(365)

    pl.plot_components()

    plt.plot(np.append(fitted['yhat'], predicted['yhat']), alpha=.3)
    plt.plot(forecast['yhat'], alpha=.3)
    plt.show()

#%%
    from ProphetLite.fourier import get_fourier_series
    exo = get_fourier_series(length=365,
                             seasonal_period=365.25,
                             fourier_order=4)
    exogenous = None
    for i in range(int(len(y) // 365.25) + 1):
        zero_exo = np.zeros((len(y), 8))
        try:
            zero_exo[i * 365: (i + 1) * 365, :] = exo
        except:
            zero_exo[i * 365: (i + 1) * 365, :] = exo[:350]
        if exogenous is None:
            exogenous = zero_exo
        else:
            exogenous = np.concatenate((exogenous, zero_exo), axis=1)

    # plt.plot(exogenous)

    pl = ProphetLite()
    fitted = pl.fit(y, [365.25], exogenous=exogenous, alpha=5) #To pass multiple seasonalities just add more nu
    predicted = pl.predict(365, exogenous=exogenous[-365:, :])


    pl.plot_components()
    pl.plot()

    plt.plot(np.append(fitted['yhat'], predicted['yhat']), alpha=.3, label='ProphetLite')
    plt.plot(forecast['yhat'], alpha=.3, label='Prophet')
    plt.legend()
    plt.show()

    plt.plot(np.append(fitted['trend'], predicted['trend']), alpha=.3, label='ProphetLite')
    plt.plot(forecast['trend'], alpha=.3, label='Prophet')
    plt.legend()
    plt.show()
