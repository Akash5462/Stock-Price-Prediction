import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from prophet import Prophet


def train_regression(df):
    """
    Train a simple Linear Regression model on Close price.
    Uses index (time steps) as X and closing price as y.
    """
    if df.empty:
        return None

    X = np.arange(len(df)).reshape(-1, 1)   # time steps
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    return model


def train_prophet(df):
    """
    Train a Prophet model on Close prices.
    Expects a DataFrame with 'Date' and 'Close' columns.
    """
    if df.empty:
        return None

    try:
        data = df[['Date', 'Close']].copy()
        data = data.rename(columns={"Date": "ds", "Close": "y"})

        # Ensure proper types
        data['ds'] = pd.to_datetime(data['ds'])
        data['y'] = pd.to_numeric(data['y'], errors='coerce')

        data = data.dropna()

        if len(data) < 30:   # Prophet needs some history
            print("Not enough data for Prophet.")
            return None

        model = Prophet()
        model.fit(data)

        return model
    except Exception as e:
        print(f"Prophet training failed: {e}")
        return None
