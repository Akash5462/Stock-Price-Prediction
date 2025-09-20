from data_loader import load_data as load_stock_data
from features import add_features
from models import train_regression, train_prophet

# Top 100 Indian stocks (NSE tickers with .NS)
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

if __name__ == "__main__":
    for ticker in tickers:
        print(f"\nTraining models for {ticker}...")
        df = load_stock_data(ticker)
        df = add_features(df)

        if df.empty:
            print(f"Skipping {ticker}, no data.")
            continue

        reg_model = train_regression(df)
        prophet_model = train_prophet(df)

        print(f"{ticker}: Models trained successfully")
