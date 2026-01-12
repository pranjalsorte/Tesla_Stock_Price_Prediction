# -----------------------------------------------
# Tesla Stock Price Prediction - Streamlit App
# Using SimpleRNN and LSTM
# -----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



# -----------------------------------------------
# App Title & Description
# -----------------------------------------------

st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide")

st.title("📈 Tesla Stock Price Prediction")
st.write("Predict Tesla stock closing price using SimpleRNN and LSTM models")

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/Tesla.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

st.subheader("📊 Tesla Stock Data")
st.write(df.tail())

# -----------------------------------------------
# Scaling the Data
# -----------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Adj Close']])

# -----------------------------------------------
# Load Models
# -----------------------------------------------

simple_rnn_model = load_model("models/simple_rnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

# -----------------------------------------------
# Create Input Sequence
# -----------------------------------------------

def create_input_sequence(data, window_size=60):
    return np.array(data[-window_size:]).reshape(1, window_size, 1)

# -----------------------------------------------
# User Input
# -----------------------------------------------

days = st.selectbox(
    "Select Prediction Horizon",
    options=[1, 5, 10]
)

# -----------------------------------------------
# Prediction Logic
# -----------------------------------------------

def predict_future(model, data, days):
    temp_data = list(data.flatten())
    predictions = []

    for _ in range(days):
        input_seq = np.array(temp_data[-60:]).reshape(1, 60, 1)
        pred = model.predict(input_seq, verbose=0)
        temp_data.append(pred[0][0])
        predictions.append(pred[0][0])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# -----------------------------------------------
# Generate Predictions
# -----------------------------------------------

if st.button("Predict Stock Price"):

    rnn_future = predict_future(simple_rnn_model, scaled_data, days)
    lstm_future = predict_future(lstm_model, scaled_data, days)

    st.subheader("🔮 Future Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### SimpleRNN Prediction")
        st.write(rnn_future)

    with col2:
        st.write("### LSTM Prediction")
        st.write(lstm_future)

    # -------------------------------------------
    # Plot Predictions
    # -------------------------------------------

    st.subheader("📉 Prediction Visualization")

    plt.figure(figsize=(10, 5))
    plt.plot(rnn_future, label="SimpleRNN Prediction")
    plt.plot(lstm_future, label="LSTM Prediction")
    plt.xlabel("Days Ahead")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(plt)
