from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import tensorflow as tf  # For loading the GRU model
import seaborn as sns
from datetime import datetime
import io
import base64

# Use the 'Agg' backend for matplotlib to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Global dictionary to cache downloaded and preprocessed data by stock symbol.
cached_data = {}

def preprocess_data(df):
    """
    Preprocesses the DataFrame:
      - Uses the last 15% of the data.
      - Converts 'Date' to datetime and numeric columns.
      - Sets the Date as the index and sorts it.
      - Performs feature engineering.
      - Scales selected features.
      - Splits the data and prints the shapes.
    """
    n = len(df)
    df = df.iloc[int(0.85 * n):].copy()  
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
    
    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set Date as index and sort
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Feature Engineering
    df.loc[:, 'Returns'] = df['Close'].pct_change()
    df.loc[:, 'Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.loc[:, 'MA5'] = df['Close'].rolling(window=5).mean()
    df.loc[:, 'MA20'] = df['Close'].rolling(window=20).mean()
    df.loc[:, 'MA50'] = df['Close'].rolling(window=50).mean()
    df.loc[:, 'Daily_Range'] = df['High'] - df['Low']
    df.loc[:, 'Volatility'] = df['Returns'].rolling(window=20).std()
    df.loc[:, 'Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df.loc[:, 'Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df.loc[:, 'MACD'] = exp1 - exp2
    df.loc[:, 'Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Price Momentum
    df.loc[:, 'Momentum'] = df['Close'] - df['Close'].shift(4)

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

    # Log the sizes of splits
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

def get_next_trading_days(last_date, num_days):
    """
    Generates the next `num_days` trading days (skipping weekends).
    """
    future_dates = []
    current_date = last_date
    while len(future_dates) < num_days:
        current_date += pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Monday=0, Tuesday=1, ... Friday=4
            future_dates.append(current_date)
    return future_dates

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

# --- Combined Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    This endpoint performs the complete workflow:
      - Downloads the data for the selected stock if not cached.
      - Preprocesses the data and caches it.
      - Prepares an input sequence from the raw 'Close' values.
      - Loads the pre-trained GRU model.
      - Predicts the next 2 days' close prices.
      - Plots:
          * Historical data in blue (with markers).
          * A single red line (no markers) from the last historical point to the last predicted point.
          * Exactly two red markers for the two predicted points.
      - Renders the predictions, plot, and warning on a new page.
    """
    selected_stock = request.form.get('stock')
    if not selected_stock:
        flash("Please select a stock.")
        return redirect(url_for('index'))
    
    try:
        # Download and cache data if not already cached
        if selected_stock not in cached_data:
            data = yf.download(selected_stock, period="max")
            print(f"Raw data for {selected_stock}:")
            print(data)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
            data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            print("Cleaned data:")
            print(data)
            print("Data head:")
            print(data.head())
            processed = preprocess_data(data)
            cached_data[selected_stock] = {"raw": data, "preprocessed": processed}
        else:
            print(f"Using cached data for {selected_stock}")
            data = cached_data[selected_stock]["raw"]
        
        # Prepare input using the raw 'Close' values
        sequence_length = 50  # Must match training sequence length
        recent_data = data['Close'].values
        if len(recent_data) < sequence_length:
            flash("Not enough data to form the prediction sequence.")
            return redirect(url_for('index'))
        
        # Compute scaling parameters (assuming MinMax scaling was used)
        close_min = data['Close'].min()
        close_max = data['Close'].max()
        scaled_recent = (recent_data - close_min) / (close_max - close_min)
        input_seq = scaled_recent[-sequence_length:].reshape(1, sequence_length, 1)
        
        # Load the pre-trained GRU model (using custom_objects to resolve 'mse')
        gru_model = tf.keras.models.load_model("models/gru_model.h5", 
                                                custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        
        def predict_next_days(model, input_sequence, num_days):
            predictions = []
            current_sequence = input_sequence.copy()
            for _ in range(num_days):
                next_pred_scaled = model.predict(current_sequence)[0][0]
                predictions.append(next_pred_scaled)
                # Append new prediction and remove the oldest value
                current_sequence = np.append(current_sequence[:, 1:, :], [[[next_pred_scaled]]], axis=1)
            predictions_actual = [(p * (close_max - close_min)) + close_min for p in predictions]
            return predictions_actual

        # Make 2-day predictions
        future_predictions = predict_next_days(gru_model, input_seq, 2)
        print("Predicted Close Prices for the Next 2 Days:", future_predictions)
        
        # --- Generate the Plot ---
        # Use the last 100 days for historical context (or fewer if not available)
        historical_data = data.tail(100).copy()
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        # Generate future trading dates (skipping weekends)
        last_date = historical_data['Date'].iloc[-1]
        future_dates = get_next_trading_days(last_date, len(future_predictions))
        
        # Identify the last historical point for continuity
        last_hist_date = historical_data['Date'].iloc[-1]
        last_hist_close = historical_data['Close'].iloc[-1]
        
        # The two predicted points
        pred_dates = future_dates
        pred_values = future_predictions
        
        plt.figure(figsize=(10, 6))
        
        # 1. Plot historical data in blue (with markers)
        plt.plot(historical_data['Date'], 
                 historical_data['Close'], 
                 label='Historical Close Price', 
                 color='blue', 
                 marker='o')
        
        # 2. Plot one red line from the last historical point to the last predicted date
        #    with NO markers to avoid a red marker on the last historical point.
        line_dates = [last_hist_date] + list(pred_dates)
        line_values = [last_hist_close] + list(pred_values)
        plt.plot(line_dates, line_values, color='red', label='Predicted Close Price', marker=None)
        
        # 3. Plot exactly TWO red markers for the two predicted points
        plt.plot(pred_dates, pred_values, color='red', marker='o', linestyle='None')
        
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title(f"Historical Close Prices and Predictions for {selected_stock}")
        plt.legend()
        
        # Convert plot to PNG and then to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_png = buf.getvalue()
        buf.close()
        plot_url = base64.b64encode(image_png).decode('utf8')
        plt.close()  # Close the figure to free memory

        # Render template with predictions, plot, and warning message
        return render_template('prediction.html', 
                               predictions=future_predictions, 
                               stock=selected_stock,
                               plot_url=plot_url)
    
    except Exception as e:
        print(f"Error in prediction for {selected_stock}: {e}")
        flash(f"Error in prediction: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
