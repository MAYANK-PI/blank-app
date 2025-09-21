# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# ----------------- Streamlit Page Config -----------------
st.set_page_config(layout="wide", page_title="📈 Stock Forecasting Dashboard")

st.title("📊 Stock Market Time Series Forecasting")
st.markdown("This app compares *ARIMA, Prophet, SARIMA, and LSTM* forecasts with actual stock prices.")

# ----------------- Helper Functions -----------------
def read_forecast_file(path):
    """Flexible reader for forecast files"""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.shape[1] == 1:
            return df.iloc[:, 0].rename(os.path.splitext(os.path.basename(path))[0])
        if "yhat" in df.columns:  # Prophet style
            return pd.Series(df["yhat"].values, index=pd.to_datetime(df["ds"]))
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return df[col].rename(os.path.splitext(os.path.basename(path))[0])
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
    return None

def load_main():
    """Load main stock dataset"""
    for fname in ["nifty50_clean.csv", "nifty50.csv"]:
        if os.path.exists(fname):
            return pd.read_csv(fname, parse_dates=["Date"], index_col="Date")
    return None

# ----------------- Load Data -----------------
df = load_main()
if df is None:
    st.error("❌ No Nifty50 data found! Place nifty50.csv or nifty50_clean.csv in the folder.")
    uploaded = st.file_uploader("Or upload a stock CSV file (with 'Date' & 'Close' columns)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["Date"], index_col="Date")
        st.success("✅ File uploaded successfully.")

if df is None:
    st.stop()

# ----------------- Show Data -----------------
st.subheader("📂 Data Preview")
st.dataframe(df.head())

st.subheader("📈 Historical Closing Price")
st.line_chart(df["Close"])

# ----------------- Load Forecasts -----------------
forecast_files = {
    "ARIMA": "arima_forecast.csv",
    "Prophet": "prophet_forecast.csv",
    "SARIMA": "sarima_forecast.csv",
    "LSTM": "lstm_forecast.csv",
}
forecasts = {}
for name, fname in forecast_files.items():
    s = read_forecast_file(fname)
    if s is not None:
        forecasts[name] = s

if not forecasts:
    st.info("⚠ No forecast files found (arima_forecast.csv, prophet_forecast.csv, etc.).")
    st.stop()

# ----------------- Model Comparison -----------------
st.subheader("🔍 Compare Forecast Models")

results = pd.DataFrame({"Actual": df["Close"]})
for name, series in forecasts.items():
    results[name] = series

results = results.dropna(how="all")

# Select models
selected = st.multiselect("Select models to compare", list(forecasts.keys()), default=list(forecasts.keys()))

# ----------------- Evaluation Metrics -----------------
st.subheader("📊 Model Performance Metrics")
metrics = []
for name in selected:
    if name in results.columns and results[name].notna().any():
        y_true = results["Actual"].dropna()
        y_pred = results[name].reindex(y_true.index).dropna()
        common_idx = y_true.index.intersection(y_pred.index)
        if len(common_idx) > 0:
            mae = mean_absolute_error(y_true.loc[common_idx], y_pred.loc[common_idx])
            rmse = sqrt(mean_squared_error(y_true.loc[common_idx], y_pred.loc[common_idx]))
            mape = np.mean(np.abs((y_true.loc[common_idx] - y_pred.loc[common_idx]) / y_true.loc[common_idx])) * 100
            metrics.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

if metrics:
    metrics_df = pd.DataFrame(metrics).set_index("Model")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"}))
else:
    st.warning("⚠ Not enough overlapping data to compute metrics.")

# ----------------- Plot Actual vs Forecast -----------------
st.subheader("📉 Actual vs Forecasted Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results.index, results["Actual"], label="Actual", color="black")
for name in selected:
    if name in results.columns:
        ax.plot(results.index, results[name], linestyle="--", label=name)
ax.set_title("Stock Price Forecast Comparison")
ax.legend()
st.pyplot(fig)

# ----------------- Download Combined Results -----------------
st.subheader("⬇ Download Results")
csv = results.to_csv().encode("utf-8")
st.download_button("Download as CSV", csv, "combined_results.csv", "text/csv")
