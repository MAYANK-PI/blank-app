# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("📈 NIFTY 50 Stock Forecasting with LSTM")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload NIFTY50 CSV (Date, Close)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date")

    st.write("### Raw Data", df.tail())

    # --- Moving Average ---
    df["MA20"] = df["Close"].rolling(20).mean()
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close Price")
    ax.plot(df["Date"], df["MA20"], label="20-Day MA")
    ax.set_title("NIFTY50 Close Price with Moving Average")
    ax.legend()
    st.pyplot(fig)

    # --- Data Preparation for LSTM ---
    data = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i+time_step), 0])
            y.append(dataset[i+time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # --- Build LSTM Model ---
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    # --- Train ---
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # --- Prediction ---
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- Plot Actual vs Prediction ---
    fig, ax = plt.subplots()
    ax.plot(df["Date"].iloc[-len(y_test):], y_test_rescaled, label="Actual")
    ax.plot(df["Date"].iloc[-len(y_test):], test_pred, label="Predicted")
    ax.set_title("Actual vs Predicted Closing Price (Test Data)")
    ax.legend()
    st.pyplot(fig)

    # --- Forecast Next 3 Days ---
    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1, -1, 1)

    future_preds = []
    input_seq = last_60.copy()

    for _ in range(3):  # predict next 3 days
        pred = model.predict(input_seq)[0][0]
        future_preds.append(pred)

        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    st.write("### Next 3 Days Forecast")
    future_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3"],
        "Predicted Close": future_preds.flatten()
    })
    st.write(future_df)
