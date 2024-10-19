import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd

# Function to fetch stock data with a date range
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data for LSTM
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Apply custom CSS for dark theme
def add_custom_css():
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: black;
        }
        .sidebar .sidebar-content {
            background-color: black;
            color: white;
        }
        .stButton>button {
            background-color: green;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black;
        }
        p {
            color: black;
        }
        .css-1lcbmhc {
            color: black;
        }
        .css-10trblm {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app
def create_app():
    add_custom_css()

    st.title('Enhanced Stock Price Prediction')
    st.write("A simple LSTM-based stock price prediction app.")

    # Sidebar options
    st.sidebar.header("User Input")
    
    # Stock ticker input
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    
    # Date range input
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    # Number of epochs input
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=50, value=10)
    
    # Button to fetch data and make prediction
    if st.sidebar.button('Predict'):
        stock_data = get_stock_data(ticker, start_date, end_date)
        st.write(f"Fetching data for {ticker} from {start_date} to {end_date}")

        # Plot historical data
        st.subheader('Historical Stock Prices')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='royalblue')))
        st.plotly_chart(fig, use_container_width=True)

        # Prepare and train the model
        X_train, y_train, X_test, y_test, scaler = prepare_data(stock_data)
        model = build_model()

        # Train model on training data
        with st.spinner('Training the model...'):
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Plot predicted vs actual prices
        st.subheader('Predicted vs Actual Stock Prices')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=stock_data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted Close Price', line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=stock_data.index[-len(predictions):], y=stock_data['Close'].values[-len(predictions):], mode='lines', name='Actual Close Price', line=dict(color='green')))
        st.plotly_chart(fig2, use_container_width=True)

        # Export predictions to CSV
        export_data = pd.DataFrame({
            'Date': stock_data.index[-len(predictions):],
            'Predicted Close Price': predictions.flatten(),
            'Actual Close Price': stock_data['Close'].values[-len(predictions):]
        })
        
        if st.sidebar.button('Export Predictions to CSV'):
            export_data.to_csv('predicted_stock_prices.csv', index=False)
            st.sidebar.success("File exported successfully!")

# Run the app
if __name__ == "__main__":
    create_app()
