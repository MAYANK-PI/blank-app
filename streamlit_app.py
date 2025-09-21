import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

st.title("Nifty 50 Stock Price Forecasting (Next 3 Days)")

# -----------------
# Get stock data
# -----------------
ticker = "^NSEI"  # Nifty 50 Index
data = yf.download(ticker, period="6mo")  # last 6 months
data = data[['Close']]
st.subheader("Latest Closing Prices")
st.line_chart(data['Close'])

# -----------------
# Preprocessing
# -----------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

prediction_days = 60  # LSTM window size
x_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -----------------
# Build LSTM model
# -----------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# -----------------
# Predict next 3 days
# -----------------
last_60_days = scaled_data[-prediction_days:]
future_input = last_60_days.reshape(1,-1)
future_input = future_input[0].tolist()

predicted_prices = []
for i in range(3):
    x_test = np.array(future_input[-prediction_days:])
    x_test = x_test.reshape(1,prediction_days,1)
    pred = model.predict(x_test)
    future_input.append(pred[0][0])
    predicted_prices.append(pred[0][0])

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))

# -----------------
# Plot actual vs predicted
# -----------------
st.subheader("Next 3 Days Forecast vs Latest Closing Price")
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=3)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices.flatten()})
st.dataframe(forecast_df)

# Plot
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Actual Closing Price')
plt.plot(pd.date_range(start=data.index[-1], periods=4)[1:], predicted_prices, label='Forecasted Price', marker='o')
plt.title("Nifty 50 Actual vs Predicted (Next 3 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)

# -----------------
# Moving Average
# -----------------
data['MA20'] = data['Close'].rolling(window=20).mean()
st.subheader("Moving Average (20 days)")
st.line_chart(data[['Close','MA20']])
