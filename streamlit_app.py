# forecast_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings("ignore")

# -------------------- STEP 1: Load Data --------------------
df = pd.read_csv("nifty50.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()
df = df[["Close"]]   # keep only Close column
print("Data loaded:", df.shape)

# -------------------- STEP 2: ARIMA --------------------
print("Training ARIMA...")
model_arima = auto_arima(df["Close"], seasonal=False, trace=False)
arima_fit = ARIMA(df["Close"], order=model_arima.order).fit()
arima_forecast = arima_fit.forecast(steps=30)
arima_forecast.to_csv("arima_forecast.csv")
print("ARIMA forecast saved.")

# -------------------- STEP 3: SARIMA --------------------
print("Training SARIMA...")
sarima_fit = SARIMAX(df["Close"], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_forecast = sarima_fit.forecast(steps=30)
sarima_forecast.to_csv("sarima_forecast.csv")
print("SARIMA forecast saved.")

# -------------------- STEP 4: Prophet --------------------
print("Training Prophet...")
prophet_df = df.reset_index().rename(columns={"Date":"ds","Close":"y"})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)[["ds","yhat"]]
prophet_forecast.to_csv("prophet_forecast.csv", index=False)
print("Prophet forecast saved.")

# -------------------- STEP 5: LSTM --------------------
print("Training LSTM (this may take a while)...")

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Prepare training sequences
train_size = int(len(scaled_data)*0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for LSTM [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Forecast next 30 days
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
i = 0
while(i < 30):
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1

lstm_forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
lstm_forecast = pd.Series(lstm_forecast.flatten(), 
                          index=pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=30))
lstm_forecast.to_csv("lstm_forecast.csv")
print("LSTM forecast saved.")

print("✅ All forecasts completed and saved (ARIMA, SARIMA, Prophet, LSTM).")
