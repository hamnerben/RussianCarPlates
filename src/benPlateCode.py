import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import get_license_plate_info_list

import csv
import numpy as np

# Load training data
plate_info_list = get_license_plate_info_list(train=True)
# plate info has valid, letter1, digits, letter2, letter3, region_code, region_name, government_info, error, price, plate_number, id

# Convert to DataFrame
data = pd.DataFrame([{
    'region': plate_info.region_name,
    'is_government': plate_info.government_info['forbidden_to_buy'],        # make these more important
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
    'id': plate_info.id,
    'region': plate_info.region_name,
    'is_government': plate_info.government_info['forbidden_to_buy'],
} for plate_info in test_plate_info_list])

# Handle categorical data (e.g., region) using one-hot encoding
test_data = pd.get_dummies(test_data, columns=['is_government'], drop_first=True)

# Align test data columns with training data columns
test_features = test_data.drop(columns=['id'])  # Exclude ID from features
new_test_data = test_features.reindex(columns=X.columns, fill_value=0)

# Predict prices
test_data['price'] = model.predict(new_test_data)

# Save predictions to CSV
test_data[['id', 'price']].to_csv('src/data/submissions/government_forest_predicted_prices.csv', index=False)  # Export only ID and price