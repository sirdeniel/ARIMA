import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error

data = pd.read_csv('~/data/stonks/AAPL_data.csv')
print(data.columns)
print(data.shape)
#data['date'] = pd.to_datetime(data['date'])
#data.set_index('date', inplace=True)
# Plotting the original Close 
#plt.figure(figsize=(14, 7))
#plt.plot(data.index, data["close"], label='Close Price')
#plt.title('Close Price Over Time')
#plt.xlabel('Date')
#plt.ylabel('Close Price')
#plt.legend()
#plt.show()

# Perform the Augmented Dickey-Fuller test on the original series
result_original = adfuller(data["close"])
print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")

if result_original[1] < 0.05:    
    print("Interpretation: The original series is Stationary.\n")
else:    
    print("Interpretation: The original series is Non-Stationary.\n")# Apply first-order differencing

data['Close_Diff'] = data['close'].diff()# Perform the Augmented Dickey-Fuller test on the differenced seriesr
result_diff = adfuller(data["Close_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")

if result_diff[1] < 0.05:    
    print("Interpretation: The differenced series is Stationary.")
else:    
    print("Interpretation: The differenced series is Non-Stationary.")


# Plotting the differenced Close price
#plt.figure(figsize=(14, 7))
#plt.plot(data.index, data['Close_Diff'], label='Differenced Close Price', color='orange')
#plt.title('Differenced Close Price Over Time')
#plt.xlabel('Date')
#plt.ylabel('Differenced Close Price')
#plt.legend()
#plt.show()

#from statsmodels.graphics.tsaplots import plot_acf, plot_pacfimport matplotlib.pyplot as plt# Plot ACF and PACF for the differenced seriesfig, axes = plt.subplots(1, 2, figsize=(16, 4))# ACF plotplot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0])axes[0].set_title('Autocorrelation Function (ACF)')# PACF plotplot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1])axes[1].set_title('Partial Autocorrelation Function (PACF)')plt.tight_layout()plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf as sm_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF for the differenced series
diff = data['Close_Diff'].dropna()
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(diff, lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(diff, lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

p = 1  # AR order used below
pacf_vals = sm_pacf(diff, nlags=max(40, p))
print(f"PACF at lag p={p}: {pacf_vals[p]:.4f}")


# Split data into train and test

train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]# Fit ARIMA model

model = ARIMA(train["close"], order=(1,1,1))

model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))# Plot the results with specified colors

plt.figure(figsize=(14,7))
plt.plot(train.index, train["close"], label='Train', color='#203147')
plt.plot(test.index, test["close"], label='Test', color='#01ef63')
plt.plot(test.index, forecast, label='Forecast', color='orange')
plt.title('Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")


forecast = forecast[:len(test)]

test_close = test["close"][:len(forecast)]# Calculate RMSE

rmse = np.sqrt(mean_squared_error(test_close, forecast))
print(f"RMSE: {rmse:.4f}")
