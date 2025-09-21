import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

st.title("Nifty 50 Stock Price Forecasting (Next 3 Days)")

# -----------------
# Upload CSV
# -----------------
uploaded_file = st.file_uploader("Upload Nifty 50 CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data.head())
    
    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        st.error("CSV must have a 'Close' column")
    else:
        data['Close'] = data['Close'].astype(float)
        st.subheader("Closing Prices")
        st.line_chart(data['Close'])
        
        # -----------------
        # Preprocessing
        # -----------------
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        prediction_days = 60
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
            x_test = x_test.reshape(1, prediction_days, 1)
            pred = model.predict(x_test)
            future_input.append(pred[0][0])
            predicted_prices.append(pred[0][0])
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))
        
        # -----------------
        # Show forecast
        # -----------------
        st.subheader("Next 3 Days Forecast")
        future_dates = pd.date_range(start=pd.to_datetime('today'), periods=3)
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices.flatten()})
        st.dataframe(forecast_df)
        
        # -----------------
        # Plot actual vs predicted
        # -----------------
        plt.figure(figsize=(10,5))
        plt.plot(data['Close'], label='Actual Closing Price')
        plt.plot(pd.date_range(start=data.index[-1], periods=4)[1:], predicted_prices, label='Forecasted Price', marker='o')
        plt.title("Actual vs Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
        
        # -----------------
        # Moving Average
        # -----------------
        data['MA20'] = data['Close'].rolling(window=20).mean()
        st.subheader("20-Day Moving Average")
        st.line_chart(data[['Close','MA20']])
