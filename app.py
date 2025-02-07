from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Form submission route for prediction (if needed)
@app.route('/predict', methods=['POST'])
def predict():
    selected_stock = request.form.get('stock')
    if selected_stock:
        flash(f"Prediction requested for stock: {selected_stock}")
    else:
        flash("Please select a stock to predict.")
    return redirect(url_for('index'))

# API endpoint: Downloads stock data using yfinance, cleans it,
# reformats the header to a single row (dropping any secondary header/ticker row),
# prints it to the console, and returns a confirmation message.
@app.route('/api/stockdata', methods=['POST'])
def stock_data():
    request_data = request.get_json()
    if not request_data or 'stock' not in request_data:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    stock_symbol = request_data['stock']
    try:
        # Download historical data (from the earliest available date up to today)
        data = yf.download(stock_symbol, period="max")
        print(f"Raw data for {stock_symbol}:")
        print(data)
        
        # If the DataFrame's columns are a MultiIndex (e.g. with a second header row for the ticker),
        # flatten the header by taking only the first level.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset the index so that the dates become a regular column.
        data.reset_index(inplace=True)
        
        # Select only the desired columns in the desired order.
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
        
        # Set a single header row (this step is optional if the header is already flat).
        data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        
        # Print the cleaned data to the console
        print("Cleaned data:")
        print(data)
        
        return jsonify({"message": f"Data for {stock_symbol} loaded and cleaned successfully. Check console for output."}), 200

    except Exception as e:
        print(f"Error loading data for {stock_symbol}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
