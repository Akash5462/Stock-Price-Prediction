from prophet import Prophet

def train_prophet(df):
    try:
        # df is expected to already have 'ds' and 'y'
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        return model
    except Exception as e:
        print(f"Prophet training failed: {e}")
        return None


def forecast_with_prophet(df, periods=180):
    """
    Returns model and forecast dataframe
    """
    try:
        model = train_prophet(df)
        if model is None:
            return None, None
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return None, None
