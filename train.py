from data_loader import load_multiple_tickers
from models import forecast_with_prophet

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

all_data = load_multiple_tickers(tickers)

# Example: Forecast RELIANCE.NS
df = all_data.get("RELIANCE.NS")
if df is not None and not df.empty:
    model, forecast = forecast_with_prophet(df, periods=180)
    if forecast is not None:
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
