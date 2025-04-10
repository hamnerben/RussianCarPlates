import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import get_license_plate_info_list

import csv
import numpy as np

# Load training data
plate_info_list = get_license_plate_info_list(train=True)

# Convert to DataFrame
data = pd.DataFrame([{
    'region': plate_info.region_name,
    'is_government': plate_info.is_government,
    'price': float(plate_info.price)
} for plate_info in plate_info_list])

# Handle categorical data (e.g., region) using one-hot encoding
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Split features and target
X = data.drop(columns=['price'])
y = data['price']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# Load test data
test_plate_info_list = get_license_plate_info_list(train=False)

# Convert test data to DataFrame
test_data = pd.DataFrame([{
    'region': plate_info.region_name,
    'is_government': plate_info.is_government
} for plate_info in test_plate_info_list])

# Handle categorical data (e.g., region) using one-hot encoding
test_data = pd.get_dummies(test_data, columns=['region'], drop_first=True)

# Align test data columns with training data columns
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict prices
test_data['predicted_price'] = model.predict(test_data)

# Save predictions to CSV
test_data.to_csv('src/data/predicted_prices.csv', index=False)