app.py:
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt  # Imported in case you need it later
import seaborn as sns
from datetime import datetime

sns.set_style("whitegrid")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

def preprocess_data(df):
    """
    Preprocesses the DataFrame:
    - Uses the last 15% of the data.
    - Converts 'Date' to datetime and numeric columns.
    - Sets the Date as the index and sorts it.
    - Performs feature engineering.
    - Scales selected features.
    - Splits the data and prints the shapes.
    Instead of plotting, it prints a confirmation message.
    """
    n = len(df)
    df = df.iloc[int(0.85 * n):]
    # df['Date'] = pd.to_datetime(df['Date'])
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

    
    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set Date as index and sort
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Range'] = df['High'] - df['Low']
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Price Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # Drop NaN values
    df.dropna(inplace=True)

    # Scale the features
    price_scaler = MinMaxScaler()
    vol_scaler = RobustScaler()
    price_features = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA20', 'MA50']
    df_scaled = df.copy()
    df_scaled[price_features] = price_scaler.fit_transform(df[price_features])
    vol_features = ['Volume', 'Volume_MA5', 'Volume_MA20', 'Volatility']
    df_scaled[vol_features] = vol_scaler.fit_transform(df[vol_features])

    # Split the data
    total_size = len(df)
    train_end = int(total_size * 0.7)
    val_end = int(total_size * 0.85)
    train_data = df_scaled[:train_end]
    val_data = df_scaled[train_end:val_end]
    test_data = df_scaled[val_end:]

    print("Preprocessing API working: Data processed successfully.")
    print("Data shape:", df.shape)
    print("Train set shape:", train_data.shape)
    print("Validation set shape:", val_data.shape)
    print("Test set shape:", test_data.shape)
    
    return df_scaled

# --- Frontend Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_stock = request.form.get('stock')
    if selected_stock:
        flash(f"Prediction requested for stock: {selected_stock}")
    else:
        flash("Please select a stock to predict.")
    return redirect(url_for('index'))

# --- API Endpoint for Stock Data and Preprocessing ---
@app.route('/api/stockdata', methods=['POST'])
def stock_data():
    request_data = request.get_json()
    if not request_data or 'stock' not in request_data:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    stock_symbol = request_data['stock']
    try:
        # Download historical data from yfinance (max available period)
        data = yf.download(stock_symbol, period="max")
        print(f"Raw data for {stock_symbol}:")
        print(data)
        
        # Flatten header if needed and clean the data
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
        data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        
        print("Cleaned data:")
        print(data)
        

        
        # Automatically trigger preprocessing; this will print its own confirmation message
        preprocess_data(data)
        print("Data head after preprocessing:")
        print(data.head())
        
        return jsonify({"message": f"Data for {stock_symbol} loaded, cleaned, and preprocessed successfully. Check console for output."}), 200
    except Exception as e:
        print(f"Error loading data for {stock_symbol}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
