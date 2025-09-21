import yfinance as yf
import pandas as pd

def load_data(ticker, period="max"):
    """
    Load data for a single ticker
    Returns a dataframe with 'ds' and 'y'
    """
    try:
        tick = yf.Ticker(ticker)
        df = tick.history(period=period)
        if df.empty:
            print(f"❌ No data available for {ticker}")
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(subset=['y'], inplace=True)
        if df.empty:
            print(f"❌ No valid closing price data for {ticker}")
            return pd.DataFrame()
        
        print(f"✅ Data loaded for {ticker} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"⚠️ Error loading {ticker}: {e}")
        return pd.DataFrame()


def load_multiple_tickers(tickers, period="max"):
    """
    Load multiple tickers and return a dictionary {ticker: df}
    """
    all_data = {}
    for ticker in tickers:
        df = load_data(ticker, period)
        if not df.empty:
            all_data[ticker] = df
    return all_data
