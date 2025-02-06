# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data fetched successfully. Total records: {len(data)}")
    return data

# Step 2: Data Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)
    
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

# Step 3: Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next stock price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Predict and Visualize
def predict_and_plot(model, X_test, y_test, scaler, original_data):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(original_data[-len(y_test):], label="True Stock Price")
    plt.plot(predictions, label="Predicted Stock Price")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.show()

# Step 5: Main Workflow
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"  # Example: Apple stock
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    look_back = 60  # Number of past days to consider
    
    # Step 1: Fetch Data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        print("No data found. Please check the ticker or date range.")
        exit()
    
    # Step 2: Preprocess Data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(stock_data)
    
    # Step 3: Build Model
    model = build_lstm_model((X_train.shape[1], 1))
    model.summary()
    
    # Step 4: Train Model
    print("Training the model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    print("Model training completed.")
    
    # Step 5: Predict and Visualize
    predict_and_plot(model, X_test, y_test, scaler, stock_data['Close'].values)
    
    # Save the Model
    model.save("stock_prediction_model.h5")
    print("Model saved as 'stock_prediction_model.h5'")
