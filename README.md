# Stock price forecasting with recurrent neural networks

My BSc Computer Science final-year project at Brunel University London (2025): a comparison
of four recurrent architectures for next-day stock price forecasting, and a Flask web app
that serves the best one.

## Model comparison

Trained on daily S&P 500 data with engineered features (returns, 5/20/50-day moving averages,
volume averages, volatility, 14-day indicators) over 50-day input sequences.
Full code and outputs: [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb).

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Simple RNN | 43.5 | 53.7 | 0.993 |
| LSTM | 74.2 | 88.4 | 0.982 |
| Bidirectional LSTM | 135.8 | 167.1 | 0.934 |
| **GRU** | **32.7** | **42.4** | **0.996** |

The GRU was best on every metric (test MAPE 0.67%) and is the model the app uses.

## The web app

Pick a stock (Apple, Alphabet, Amazon, Microsoft or the S&P 500 index). The app downloads
its full price history live from Yahoo Finance, applies the same feature engineering and
scaling, runs the trained GRU (`models/gru_model.h5`) and plots the forecast for the next
trading days, skipping weekends.

```bash
python -m venv env
env\Scripts\activate            # source env/bin/activate on macOS/Linux
pip install -r requirements.txt
python app.py                   # http://127.0.0.1:5000
```

## Stack

Python · TensorFlow/Keras · scikit-learn · pandas · yfinance · Flask · Jinja · matplotlib/seaborn

*Educational project. Not financial advice.*
