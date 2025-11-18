# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

file_name = 'tsla_2014_2023.csv'
data = pd.read_csv(file_name)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

def arima_model(data, target_variable, order):
    target_series = data[target_variable]
    train_size = int(len(target_series) * 0.8)
    train_data, test_data = target_series[:train_size], target_series[train_size:]
    
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit(method='statespace')
    forecast = fitted_model.forecast(steps=len(test_data))
    
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable + ' Price')
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()
    print("Root Mean Squared Error (RMSE):", rmse)

arima_model(data, 'close', order=(5,1,0))
```
### OUTPUT:
<img width="1727" height="1137" alt="Screenshot 2025-11-14 223037" src="https://github.com/user-attachments/assets/6b8884ce-b9ed-4fa9-a2f6-01bf328c2a93" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.

