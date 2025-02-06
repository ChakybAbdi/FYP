from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import yfinance as yf

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

# Form submission route for stock prediction (example only)
@app.route('/predict', methods=['POST'])
def predict():
    selected_stock = request.form.get('stock')
    if selected_stock:
        # Here, integrate your stock prediction logic.
        flash(f"Prediction requested for stock: {selected_stock}")
    else:
        flash("Please select a stock to predict.")
    return redirect(url_for('index'))

# API endpoint: Downloads stock data and prints it to the console
@app.route('/api/stockdata', methods=['POST'])
def stock_data():
    request_data = request.get_json()
    if not request_data or 'stock' not in request_data:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    stock_symbol = request_data['stock']
    try:
        # Download historical data for the stock (from the earliest available date up to today)
        data = yf.download(stock_symbol, period="max")
        
        # Print the downloaded data to the console for debugging
        print(f"Data for {stock_symbol}:")
        print(data)
        
        # Return a confirmation message
        return jsonify({"message": f"Data for {stock_symbol} loaded successfully and printed to the console."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
