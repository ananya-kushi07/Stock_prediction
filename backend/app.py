from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os
import io
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with React frontend

# Load the pre-trained model
model_path = os.path.join(os.getcwd(), 'AAPL_stock_prediction_model.h5')
try:
    prediction_model = load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    raise FileNotFoundError(f"Error loading the model: {e}")

@app.route('/')
def home():
    return "Welcome to the Backend!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the ticker symbol from the request
        data = request.json
        ticker = data.get('ticker', 'AAPL')  # Default to 'AAPL' if not provided

        # Fetch stock data using yfinance
        stock_data = yf.download(ticker, period="6mo")  # Fetch last 6 months of data
        if stock_data.empty:
            return jsonify({"error": "Invalid ticker symbol or no data available"}), 400

        # Process stock data
        close_prices = stock_data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Prepare the input for prediction
        sequence_length = 60
        if len(scaled_data) < sequence_length:
            return jsonify({"error": "Not enough data for prediction"}), 400

        X_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

        # Predict stock price
        prediction = prediction_model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(prediction)[0][0])

        # Compute Metrics (Assume we compare with the last real value in the dataset)
        real_price = close_prices.values[-1][0]
        mse = mean_squared_error([real_price], [predicted_price])

        # Generate Graph
        plt.figure(figsize=(10, 6))
        plt.plot(close_prices.index, close_prices.values, label='Actual Prices', color='blue')
        plt.axhline(y=predicted_price, color='red', linestyle='--', label='Predicted Price')
        plt.title(f"Stock Prices for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Send graph as a response
        response = jsonify({
            "predicted_price": predicted_price,
            "real_price": real_price,
            "mse": mse
        })
        response.headers['Content-Type'] = 'application/json'

        # Save the graph temporarily
        graph_path = os.path.join(os.getcwd(), "static", f"{ticker}_prediction_graph.png")
        plt.savefig(graph_path)
        plt.close()

        # Return response and graph URL
        return jsonify({
            "predicted_price": predicted_price,
            "real_price": real_price,
            "mse": mse,
            "graph_url": f"http://127.0.0.1:5000/static/{ticker}_prediction_graph.png"
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


@app.route('/static/<filename>', methods=['GET'])
def serve_static_file(filename):
    static_folder = os.path.join(os.getcwd(), 'static')
    file_path = os.path.join(static_folder, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
