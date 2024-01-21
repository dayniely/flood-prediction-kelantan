import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Additional imports for other models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv('D:\\Users\\Documents\\Python\\Flood Prediction\\Dataset\\transformed_data.csv')

# Drop the 'Location' column
data_model = data.drop(columns=['Location'])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_model)

# Prepare the data for the models
X = scaled_data[:-1, 1:]  # all rows except the last one, and all columns except the first (Year)
y = scaled_data[1:, 1:]   # all rows starting from second, and all columns except the first (Year)

# Reshape X for LSTM
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# ANN model
ann_model = Sequential()
ann_model.add(Dense(units=100, activation='relu', input_dim=X_train.shape[1]))
ann_model.add(Dense(units=100, activation='relu'))
ann_model.add(Dense(units=y_train.shape[1]))
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
ann_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(units=y_train_lstm.shape[1]))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# Initialize additional models
additional_models = {
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor()
}

# Train and evaluate additional models
model_performance = {}
all_models = {"ANN": ann_model, "LSTM": lstm_model, **additional_models}

for model_name, model in all_models.items():
    if model_name in ["ANN", "LSTM"]:
        # ANN and LSTM models
        predictions = model.predict(X_test if model_name == "ANN" else X_test_lstm)
        if model_name == "LSTM":
            predictions = predictions.reshape(predictions.shape[0], predictions.shape[-1])
        true_values = y_test if model_name == "ANN" else y_test_lstm
    else:
        # Other regression models
        model.fit(X_train, y_train[:, 0])
        predictions = model.predict(X_test)
        true_values = y_test[:, 0]

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions) * 100  # Convert to percentage

    model_performance[model_name] = {
        "MSE": mse,
        "MAE": mae,
        "R-squared (%)": r2
    }

# Print model performance
for model_name, metrics in model_performance.items():
    print(f"{model_name} Performance:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()
