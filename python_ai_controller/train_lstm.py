import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

print("1. Loading data...")
df = pd.read_csv(r'C:\Users\dhiya\OneDrive\Desktop\CCS_proj\ccs_training_data.csv')
data = df[['Valve_Cmd', 'Temp_Response']].values

print("2. Scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print("3. Creating Time Sequences...")
sequence_length = 60

X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    # Feed ONLY the Valve Command (Column 0) into the input X
    X.append(scaled_data[i-sequence_length:i, 0])
    # The target y is the Temperature Response (Column 1)
    y.append(scaled_data[i, 1]) 

X, y = np.array(X), np.array(y)

# LSTMs require 3D input arrays: (Samples, TimeSteps, Features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into Training (80%) and Testing (20%) sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

print("4. Building the LSTM Neural Network...")
model = Sequential()
# A lightweight LSTM with 32 neurons (Fast to train on a laptop)
model.add(LSTM(units=32, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1)) # 1 Output: The predicted temperature

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

print("5. Training the AI (This might take a minute or two)...")
# We will train for 10 "epochs" (passes over the data)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

print("6. Testing the AI on unseen data...")
predictions = model.predict(X_test)

# Un-scale the data to get real temperatures back
predictions_unscaled = predictions * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]
y_test_unscaled = y_test * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]

print("7. Plotting the Results...")
plt.figure(figsize=(12, 6))
# Plot a 500-minute slice of the test data to see it clearly
slice_end = 500
plt.plot(y_test_unscaled[:slice_end], color='blue', label='Actual Plant Physics', linewidth=2)
plt.plot(predictions_unscaled[:slice_end], color='orange', linestyle='--', label='AI Prediction', linewidth=2)
plt.title('LSTM Neural Network vs Actual Carbon Capture Plant')
plt.xlabel('Time (Minutes)')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Save the brain so we can use it tomorrow in Phase 3
model.save(r'C:\Users\dhiya\OneDrive\Desktop\CCS_proj\ccs_lstm_brain.keras')
print("Model successfully saved as 'ccs_lstm_brain.keras'. Phase 2 Complete!")
