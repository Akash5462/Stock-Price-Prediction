# 📈 Stock Price Prediction (India)

This project predicts stock prices for the **Top 100 NSE stocks** using regression and Prophet models.

## 🚀 Project Structure
- `data_loader.py` → Fetch stock data
- `features.py` → Add technical indicators
- `models.py` → ML models (Regression, Prophet)
- `train.py` → Train models on stocks
- `evaluate.py` → Evaluate performance
- `app.py` → Streamlit dashboard
- `requirements.txt` → Dependencies

## ⚡ How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Run dashboard
streamlit run app.py
