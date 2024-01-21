import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from math import sqrt

# Load the dataset
data = pd.read_csv('Dataset/transformed_data.csv')

# Store 'Location' column in a separate variable
location_data = data['Location']

# Drop the 'Location' column and normalize the data
data_model = data.drop(columns=['Location'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_model)

# Prepare the data for the LSTM model
X_lstm = scaled_data[:-1, 1:]  # all rows except the last one, and all columns except the first (Year)
y_lstm = scaled_data[1:, 1:]   # all rows starting from second, and all columns except the first (Year)

# Reshape X for LSTM
X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))

# Train-test split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# LSTM model with increased complexity and dropout layers
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=100, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=y_train_lstm.shape[1]))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the LSTM model with early stopping
history = lstm_model.fit(
    X_train_lstm, y_train_lstm, epochs=100, batch_size=32, verbose=1,
    validation_split=0.2, callbacks=[early_stopping]
)

# Predict using the trained LSTM model
lstm_predictions = lstm_model.predict(X_lstm)

# Rescale the predictions back to the original scale
lstm_predictions_rescaled = scaler.inverse_transform(np.hstack((scaled_data[:-1, :1], lstm_predictions)))

# Create a DataFrame for the predicted data
predicted_data = pd.DataFrame(lstm_predictions_rescaled, columns=data_model.columns)

# Add the 'Location' column back to the predicted data
predicted_data['Location'] = location_data[:-1]  # Exclude the last entry as it's not included in the predictions

# Reorder columns to have 'Location' as the first column
predicted_data = predicted_data[['Location'] + [col for col in predicted_data.columns if col != 'Location']]

# Save the predicted data to a new CSV file
predicted_data.to_csv('Dataset/predicted_data.csv', index=False)

# Predict on the test set
y_test_pred_lstm = lstm_model.predict(X_test_lstm)

# Rescale the predictions back to the original scale
# The test data needs to be reshaped back to the 2D shape to be inverse transformed by scaler
y_test_pred_rescaled = scaler.inverse_transform(np.hstack((scaled_data[:len(y_test_pred_lstm), :1], y_test_pred_lstm)))
y_test_actual_rescaled = scaler.inverse_transform(np.hstack((scaled_data[:len(y_test_lstm), :1], y_test_lstm)))

# Calculate RMSE
rmse = sqrt(mean_squared_error(y_test_actual_rescaled[:, 1:], y_test_pred_rescaled[:, 1:]))

# Print the RMSE
print(f"Test RMSE: {rmse:.3f}")
