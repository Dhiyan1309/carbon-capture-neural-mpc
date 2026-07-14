import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

print("Initializing AI Training Sequence...")

try:
    df = pd.read_csv('carbon_capture_data.csv')
    print("Dataset successfully loaded.")
except FileNotFoundError:
    print("ERROR: carbon_capture_data.csv not found. Run MATLAB script first to generate data.")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_temp = scaler.fit_transform(df[['Temperature']].values)

lookback = 10 
X, y = [], []

for i in range(lookback, len(scaled_temp)):
    X.append(scaled_temp[i-lookback:i, 0])
    y.append(scaled_temp[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 

print("Building LSTM Digital Twin...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training AI on physical plant dynamics...")
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

model.save('lstm_model.h5')
print("SUCCESS: Predictive Digital Twin saved as lstm_model.h5")
