import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to stock data.
    Input:
        df: DataFrame with at least ['Date', 'Close']
    Output:
        df: DataFrame with added features
    """
    if df.empty:
        return df

    df = df.copy()

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Daily Returns
    df['Returns'] = df['Close'].pct_change()

    # Volatility (Rolling Standard Deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop NaN values created by rolling windows
    df = df.dropna().reset_index(drop=True)

    return df
