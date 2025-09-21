import streamlit as st
import plotly.express as px
from data_loader import load_data
from models import forecast_with_prophet

# List of tickers
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
st.title("📈 Indian Stock Price Prediction App")

# Stock selector
ticker = st.selectbox("Choose a Stock", tickers)

# Forecast horizon selector
horizon = st.radio("Forecast Horizon:", ["6 Months", "1 Year"])
periods = 180 if horizon == "6 Months" else 365

# Load stock data (single ticker)
df = load_data(ticker)

if df is not None and not df.empty:
    st.subheader(f"📊 Historical Data for {ticker}")
    fig_hist = px.line(df, x="ds", y="y", title=f"{ticker} Stock Price History")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Prophet forecast
    st.subheader(f"🔮 Forecast for {horizon}")
    model, forecast = forecast_with_prophet(df, periods=periods)

    if forecast is not None:
        fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"{ticker} {horizon} Forecast")
        fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound")
        fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.write("📄 Predicted Prices:")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))
    else:
        st.error("❌ Forecasting failed. Not enough data.")
else:
    st.error("❌ No data available for this stock.")
