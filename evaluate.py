from sklearn.metrics import mean_squared_error

def evaluate_model(model, X, y):
    """Evaluate regression model using RMSE."""
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    return rmse
