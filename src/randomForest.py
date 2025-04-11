import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import get_license_plate_info_list
from datetime import datetime

import csv
import numpy as np

# Define SMAPE function
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

# Load training data
plate_info_list = get_license_plate_info_list(train=True)

print("lets go!")
# Convert to DataFrame
data = pd.DataFrame([{
    'id': plate_info.id,
    'region': plate_info.region_name,
    'is_government': plate_info.is_government_vehicle,  # Include government status
    'road_advantage': plate_info.government_info['road_advantage'],  # Include road advantage
    'significance_level': plate_info.government_info['significance_level'],  # Include significance level
    'price': float(plate_info.price),
    'region_code': plate_info.region_code,  # Include region code for potential future use
    'plate_digits': plate_info.digits,  # Include digits for potential future use
    'plate_length': len(plate_info.plate_number),  # Include plate length for potential future use
    'year': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').year,  # Extract year
    'month': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').month,  # Extract month
    'day_of_week': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').weekday(),  # Extract day of the week
    'day_of_year': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').timetuple().tm_yday,  # Extract day of the year
} for plate_info in plate_info_list])

# Handle categorical data (e.g., region) using one-hot encoding
data = pd.get_dummies(data, columns=['region'], drop_first=True)
print("checkpt 1")
# Split features and target
X = data.drop(columns=['price', 'id'])  # Exclude price and ID from features
y = data['price']

print("checkpt 2")
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# print("checkpt 3")
# # Perform grid search
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
# print("checkpt 3.1")
# grid_search.fit(X_train, y_train)

# print("checkpt 4")
# # Use the best model
# model = grid_search.best_estimator_

# print("checkpt 5")
# # Get feature importance
# feature_importances = model.feature_importances_
# feature_names = X.columns
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
# print(importance_df)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
smape_score = smape(y_val, y_pred)  # Calculate SMAPE

# Print validation metrics
print(f"Mean Squared Error (Region + Government Model): {mse}")
print(f"SMAPE (Region + Government Model): {smape_score}%")

# Load test data
test_plate_info_list = get_license_plate_info_list(train=False)

# Convert test data to DataFrame
test_data = pd.DataFrame([{
    'id': plate_info.id,
    'region': plate_info.region_name,
    'is_government': plate_info.is_government_vehicle,  # Include government status
    'road_advantage': plate_info.government_info['road_advantage'],  # Include road advantage
    'significance_level': plate_info.government_info['significance_level'],  # Include significance level
    'region_code': plate_info.region_code,  # Include region code for potential future use
    'plate_digits': plate_info.digits,  # Include digits for potential future use
    'plate_length': len(plate_info.plate_number),  # Include plate length for potential future use
    'year': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').year,  # Extract year
    'month': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').month,  # Extract month
    'day_of_week': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').weekday(),  # Extract day of the week
    'day_of_year': datetime.strptime(plate_info.date, '%Y-%m-%d %H:%M:%S').timetuple().tm_yday,  # Extract day of the year
} for plate_info in test_plate_info_list])

# Handle categorical data (e.g., region) using one-hot encoding
test_data = pd.get_dummies(test_data, columns=['region'], drop_first=True)

# Align test data columns with training data columns
test_features = test_data.drop(columns=['id'])  # Exclude ID from features
test_features = test_features.reindex(columns=X.columns, fill_value=0)  # Align with training data columns

# # Scale numerical features
# scaler = StandardScaler()
# data[['road_advantage', 'significance_level']] = scaler.fit_transform(data[['road_advantage', 'significance_level']])
# test_data[['road_advantage', 'significance_level']] = scaler.transform(test_data[['road_advantage', 'significance_level']])

# Predict prices
test_data['price'] = model.predict(test_features)

# Save predictions to CSV
test_data[['id', 'price']].to_csv('src/data/submissions/date_forest_predicted_prices.csv', index=False)  # Export only ID and price