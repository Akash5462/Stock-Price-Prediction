import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data
from features import add_features
from models import train_regression, train_prophet

st.set_page_config(page_title="Stock Price Prediction India", layout="wide")

st.title("📈 Stock Price Prediction App (India)")

# ---------------------- Ticker List ----------------------
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ADANIENT.NS", "JSWSTEEL.NS",
    "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "TECHM.NS", "COALINDIA.NS",
    "GRASIM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "DIVISLAB.NS", "NESTLEIND.NS",
    "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS",
    "BPCL.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "SBILIFE.NS", "HAVELLS.NS",
    "PIDILITIND.NS", "DABUR.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "TORNTPHARM.NS",
    "GODREJCP.NS", "COLPAL.NS", "AMBUJACEM.NS", "SHREECEM.NS", "INDUSINDBK.NS",
    "VOLTAS.NS", "UPL.NS", "BIOCON.NS", "PEL.NS", "BANDHANBNK.NS",
    "LICI.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "DMART.NS",
    "IRCTC.NS", "POLYCAB.NS", "ABB.NS", "SIEMENS.NS", "GAIL.NS",
    "IOC.NS", "HINDPETRO.NS", "MANAPPURAM.NS", "CANBK.NS", "PNB.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "YESBANK.NS", "BANKBARODA.NS", "UNIONBANK.NS",
    "BHEL.NS", "BEL.NS", "MOTHERSON.NS", "ASHOKLEY.NS", "TATAMOTORS.NS",
    "TVSMOTOR.NS", "APOLLOTYRE.NS", "JKTYRE.NS", "INDIGO.NS", "SPICEJET.NS",
    "ZEEL.NS", "PVRINOX.NS", "BALRAMCHIN.NS", "RPOWER.NS", "INDUSTOWER.NS",
    "HAL.NS", "BOSCHLTD.NS", "TRENT.NS", "LTIM.NS", "MINDTREE.NS"
]

# Sidebar: select ticker
ticker = st.sidebar.selectbox("Select Stock Ticker", options=tickers, index=0)

# Date range
start = "2015-01-01"
end = "2025-12-31"

# ---------------------- Run Prediction ----------------------
if st.button("Run Prediction"):
    df = load_data(ticker, start, end)

    if df is None or df.empty:
        st.error("❌ Failed to fetch stock data. Try another ticker.")
    else:
        st.success(f"✅ Models trained for {ticker}")

        # Feature engineering
        df = add_features(df)

        # ---------------- Linear Regression ----------------
        reg_model = train_regression(df)
        if reg_model:
            X_future = np.arange(len(df) + 30).reshape(-1, 1)  # next 30 days
            y_pred = reg_model.predict(X_future)

            plt.figure(figsize=(12, 6))
            plt.plot(df['Date'], df['Close'], label="Actual", color="blue")
            plt.plot(pd.date_range(df['Date'].iloc[0], periods=len(y_pred), freq="D"),
                     y_pred, label="Linear Regression", linestyle="--", color="orange")
            plt.title(f"{ticker} - Linear Regression Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

        # ---------------- Prophet ----------------
        prophet_model = train_prophet(df)
        if prophet_model:
            future = prophet_model.make_future_dataframe(periods=30)
            forecast = prophet_model.predict(future)

            fig2 = prophet_model.plot(forecast)
            st.pyplot(fig2)

            fig3 = prophet_model.plot_components(forecast)
            st.pyplot(fig3)
