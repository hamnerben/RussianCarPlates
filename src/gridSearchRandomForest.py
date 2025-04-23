import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import get_license_plate_info_list

import numpy as np

# Define SMAPE function
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# Use make_scorer to create a scoring object for GridSearchCV
smape_scorer = make_scorer(smape, greater_is_better=False)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],  # Number of trees, commonly used values.
    'max_depth': [10, 20, None],  # Depth of the trees; None means trees are expanded until pure.
    'min_samples_split': [2, 10],  # Minimum number of samples required to split an internal node.
    'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node.
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees.
}


# Load training data
plate_info_list = get_license_plate_info_list(train=True)
print("Training data loaded successfully!")

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
} for plate_info in plate_info_list])

print("DataFrame created successfully with all required features.")

# Handle categorical data (e.g., region) using one-hot encoding
data = pd.get_dummies(data, columns=['region'], drop_first=True)
print("Categorical variables encoded using one-hot encoding.")

# Split features and target
X = data.drop(columns=['price', 'id'])  # Exclude price and ID from features
y = data['price']
print("Data split into features (X) and target (y).")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training and validation data split (80-20).")

# Perform grid search
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring=smape_scorer
)
print("Starting Grid Search for hyperparameter tuning.")

grid_search.fit(X_train, y_train)
print("Grid Search completed. Best parameters found.")
print("Best hyperparameters:", grid_search.best_params_)


# Use the best model
model = grid_search.best_estimator_
print("Best model selected based on grid search.")

# Get feature importance
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print("Feature importances calculated and sorted:")
print(importance_df)

# Validate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
smape_score = smape(y_val, y_pred) 

# Print validation metrics
print(f"Validation Metrics:")
print(f"Mean Squared Error (Region + Government Model): {mse}")
print(f"SMAPE (Region + Government Model): {smape_score}%")

# Load test data
test_plate_info_list = get_license_plate_info_list(train=False)
print("Test data loaded successfully!")

# Convert test data to DataFrame
test_data = pd.DataFrame([{
    'id': plate_info.id,
    'region': plate_info.region_name,
    'is_government': plate_info.is_government_vehicle,  
    'road_advantage': plate_info.government_info['road_advantage'],  
    'significance_level': plate_info.government_info['significance_level'],  
    'region_code': plate_info.region_code,  
    'plate_digits': plate_info.digits,  
    'plate_length': len(plate_info.plate_number),  
} for plate_info in test_plate_info_list])

print("Test data converted to DataFrame.")

# Handle categorical data (e.g., region) using one-hot encoding
test_data = pd.get_dummies(test_data, columns=['region'], drop_first=True)
print("Test data categorical variables encoded using one-hot encoding.")

# Align test data columns with training data columns
test_features = test_data.drop(columns=['id'])  
test_features = test_features.reindex(columns=X.columns, fill_value=0)  
print("Test data columns aligned with training data columns.")

# Scale numerical features
scaler = StandardScaler()
data[['road_advantage', 'significance_level']] = scaler.fit_transform(data[['road_advantage', 'significance_level']])
test_data[['road_advantage', 'significance_level']] = scaler.transform(test_data[['road_advantage', 'significance_level']])
print("Numerical features scaled using StandardScaler.")

# Predict prices
test_data['price'] = model.predict(test_features)
print("Predictions for test data completed.")

# Save predictions to CSV
test_data[['id', 'price']].to_csv('src/data/submissions/scaled_forest_predicted_prices.csv', index=False)
print("Predictions saved to CSV file: scaled_forest_predicted_prices.csv")
