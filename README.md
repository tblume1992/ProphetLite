# ProphetLite
 Like Prophet but in numba and all fit with LASSO

## A basic comparison vs Prophet:
```
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
```
## Now for ProphetLite
 You must pass your data as a numpy array
```
    y = df['y'].values
```
 Now to build the class and pass the seasonality
```
pl = ProphetLite()
fitted = pl.fit(y, [365.25]) #To pass multiple seasonalities just add more nu
predicted = pl.predict(365)

    pl.plot()
    pl.plot_components()

    plt.plot(np.append(fitted['yhat'], predicted['yhat']), alpha=.3)
    plt.plot(forecast['yhat'], alpha=.3)
    plt.show()

    plt.plot(np.append(fitted['trend'], predicted['trend']))
    plt.plot(forecast['trend'])
    plt.show()