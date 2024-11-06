### Developed by : SAFEEQ FAZIL A
### Register by : 212222240086
### Date:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
file_path = '/content/vegetable.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Select a specific commodity (for example, 'Tomato Big(Nepali)')
commodity = 'Tomato Big(Nepali)'
commodity_data = data[data['Commodity'] == commodity]

# Set the Date column as the index and resample to monthly average
commodity_data.set_index('Date', inplace=True)
monthly_data = commodity_data['Average'].resample('M').mean()

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(monthly_data, label=f'Monthly Average Price of {commodity}')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'Time Series of {commodity} Prices')
plt.legend()
plt.show()

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(monthly_data.dropna(), lags=24, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(212)
plot_pacf(monthly_data.dropna(), lags=24, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets (last 12 months as test data)
train_data = monthly_data[:-12]
test_data = monthly_data[-12:]

# Fit the SARIMA model on the training data
# SARIMA model order: (p, d, q) x (P, D, Q, s) where s is the seasonality period (12 for monthly data)
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast the next 12 months
forecast = sarima_result.get_forecast(steps=12)
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# **Handling NaN values before calculating RMSE**
# 1. Align the forecast index with the test data index
forecast_values = forecast_values.reindex(test_data.index)  
# 2. Remove rows with NaN values from both test_data and forecast_values to ensure alignment
#     This is done using the 'inner' join in pd.concat, which keeps only the common index values.
valid_data = pd.concat([test_data, forecast_values], axis=1, join='inner').dropna()  
test_data_valid = valid_data.iloc[:, 0] # Extract the valid test data
forecast_values_valid = valid_data.iloc[:, 1] # Extract the corresponding valid forecast values


# Calculate the Root Mean Squared Error (RMSE) using valid data
rmse = sqrt(mean_squared_error(test_data_valid, forecast_values_valid))
print(f'Root Mean Squared Error: {rmse}')

# Plot the forecast along with training and test data
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data', color='orange')
plt.plot(forecast_values, label='Forecasted Data', color='red')
plt.fill_between(forecast_values.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'SARIMA Forecasting for {commodity} with RMSE: {rmse:.2f}')
plt.legend()
plt.show()

# Print forecasted values
print("Forecasted values for the next 12 months:")

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/d3832c5f-6ee3-462a-bf26-eaac8f3eb8d7)
![Screenshot 2024-11-06 092743](https://github.com/user-attachments/assets/ae086b3d-3952-4601-9a7c-256c183d4b81)


![image](https://github.com/user-attachments/assets/131d9d28-f1c7-4103-8446-db80cc88c225)


### RESULT:
Thus the program run successfully based on the SARIMA model.
