import yfinance as yf
import pandas as pd

def load_data(ticker, start="2015-01-01", end="2025-12-31"):
    """
    Fetch stock data for the given ticker from Yahoo Finance.
    Returns a clean DataFrame with Date, OHLC, and Volume.
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)

        # Rename index column to Date if needed
        if "Date" not in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)

        # Define preferred columns
        cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        # Keep only those available in df
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return pd.DataFrame()
