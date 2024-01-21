import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('D:\\Users\\Documents\\Python\\Flood Prediction\\Dataset\\transformed_data.csv')

# Assuming the first column after 'Location' and 'Year' is the target
target_column = data.columns[2]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.drop(columns=['Location', 'Year']))

# Prepare the data for the models
X = scaled_data[:-1]  # all rows except the last one
y = scaled_data[1:, 0]  # all rows starting from second, target is the first column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVR and LR models
svr_model = SVR()
lr_model = LinearRegression()

# Train SVR and LR models
svr_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Predict with SVR and LR models
svr_predictions = svr_model.predict(X)
lr_predictions = lr_model.predict(X)

# Evaluate models
svr_mse = mean_squared_error(y, svr_predictions)
lr_mse = mean_squared_error(y, lr_predictions)
print(f"SVR MSE: {svr_mse:.4f}")
print(f"LR MSE: {lr_mse:.4f}")

# Create predicted datasets, excluding the first row which was not used for predictions
data_svr = data.iloc[1:].copy()  # Exclude the first row
data_lr = data.iloc[1:].copy()  # Exclude the first row

# Replace the target column with the predictions
data_svr[target_column] = svr_predictions
data_lr[target_column] = lr_predictions

# Save the predicted data to individual CSV files
data_svr.to_csv('D:\\Users\\Documents\\Python\\Flood Prediction\\Dataset\\svr_predicted_data.csv', index=False)
data_lr.to_csv('D:\\Users\\Documents\\Python\\Flood Prediction\\Dataset\\lr_predicted_data.csv', index=False)
