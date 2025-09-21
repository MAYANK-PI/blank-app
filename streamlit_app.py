# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Try imports safely ---
try:
    from pmdarima import auto_arima
    PMDARIMA = True
except Exception:
    PMDARIMA = False

try:
    from prophet import Prophet
    PROPHET = True
except Exception:
    PROPHET = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA = True
except Exception:
    SARIMA = False

# --- Load Data ---
st.title("📈 NIFTY50 Stock Price Forecasting with LSTM")

uploaded_file = st.file_uploader("Upload your stock CSV (with Date & Close)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna()
    df = df.sort_values("Date")
    st.write("### Raw Data", df.head())

    # Plot original series
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close Price")
    ax.set_title("NIFTY50 Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    st.pyplot(fig)

    # --- Model Selection ---
    st.sidebar.header("Select Forecasting Model")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        ["ARIMA (auto)", "SARIMA", "Prophet"]
    )

    horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 90)

    # --- Forecasting ---
    if model_choice == "ARIMA (auto)":
        if PMDARIMA:
            st.write("### Forecast with Auto ARIMA")
            series = df.set_index("Date")["Close"]

            model = auto_arima(series, seasonal=False, stepwise=True)
            forecast = model.predict(n_periods=horizon)

            future_dates = pd.date_range(df["Date"].iloc[-1], periods=horizon+1, freq="D")[1:]
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})

            fig, ax = plt.subplots()
            ax.plot(df["Date"], df["Close"], label="History")
            ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="ARIMA Forecast")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("pmdarima not available in this environment.")

    elif model_choice == "SARIMA":
        if SARIMA:
            st.write("### Forecast with SARIMA")
            series = df.set_index("Date")["Close"]

            # Example order, should tune
            model = SARIMAX(series, order=(5,1,0), seasonal_order=(1,1,1,12))
            results = model.fit(disp=False)

            forecast = results.get_forecast(steps=horizon)
            forecast_df = forecast.summary_frame()

            fig, ax = plt.subplots()
            ax.plot(series.index, series, label="History")
            ax.plot(forecast_df.index, forecast_df["mean"], label="SARIMA Forecast")
            ax.fill_between(forecast_df.index,
                            forecast_df["mean_ci_lower"],
                            forecast_df["mean_ci_upper"],
                            color="k", alpha=0.1)
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("SARIMA (statsmodels) not available.")

    elif model_choice == "Prophet":
        if PROPHET:
            st.write("### Forecast with Prophet")
            prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})

            model = Prophet(daily_seasonality=True)
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            fig = model.plot(forecast)
            st.pyplot(fig)
        else:
            st.error("Prophet not available in this environment.")
