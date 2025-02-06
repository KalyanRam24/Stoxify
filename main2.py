from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# Load pre-trained model
model = load_model("stock_prediction_model.h5")

# Initialize Flask app
app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Predict Future Stock Price
@app.route("/predict", methods=["POST"])
def predict():
    # Get stock ticker and date range from form
    ticker = request.form["ticker"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    
    # Parse the end date into a datetime object
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return render_template("result.html", error="No data found for the given ticker or date range.")
        
        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

        # Prepare input data
        look_back = 60
        
        # Ensure enough data for prediction
        if len(scaled_data) < look_back:
            return render_template("result.html", error="Not enough data to make predictions. Please select a longer date range.")
        
        # Get the last 60 data points for prediction
        X_input = scaled_data[-look_back:]
        X_input = np.reshape(X_input, (1, look_back, 1))

        # Predict future stock prices (next 'n' days)
        future_days = 10  # Predict 10 future days for example
        future_predictions = []

        for _ in range(future_days):
            predicted_price = model.predict(X_input)
            future_predictions.append(predicted_price[0][0])
            
            # Reshape the predicted price to match the 3D shape of X_input
            predicted_price = predicted_price.reshape(1, 1, 1)  # Reshape to (1, 1, 1)
            
            # Update input for next prediction
            X_input = np.append(X_input[:, 1:, :], predicted_price, axis=1)

        # Rescale predictions back to original range
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create list of future dates to display with predictions
        future_dates = [end_date + datetime.timedelta(days=i+1) for i in range(future_days)]
        future_predictions = future_predictions.flatten()

        # Return results
        return render_template(
            "result.html",
            ticker=ticker.upper(),
            predictions=list(zip(future_dates, future_predictions)),
        )
    except Exception as e:
        return render_template("result.html", error=str(e))

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1999)
