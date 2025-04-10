import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from utils import get_license_plate_info_list
import numpy as np

# Load training data
print("Loading training data...")
plate_info_list = get_license_plate_info_list(train=True)

# Convert to DataFrame
print("Converting training data to DataFrame...")
data = pd.DataFrame([{
    'region': plate_info.region_name,
    'is_government': plate_info.is_government_vehicle,  # 1 for True, 0 for False
    'gov_description': plate_info.government_info['description'],  # Will be label encoded
    'gov_forbidden_to_buy': int(plate_info.government_info['forbidden_to_buy']),  # 1 for True, 0 for False
    'gov_road_advantage': int(plate_info.government_info['road_advantage']),  # 1 for True, 0 for False
    'gov_significance_level': plate_info.government_info['significance_level'],  # Numeric
    'price': float(plate_info.price),  # Target value
} for plate_info in plate_info_list])

# Handle categorical data using LabelEncoder and One-Hot Encoding
print("Encoding categorical features...")
label_encoder = LabelEncoder()

# Label encode the 'gov_description' for the training data
data['gov_description'] = label_encoder.fit_transform(data['gov_description'])

# One-hot encode the 'region' column
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Split features and target
X = data.drop(columns=['price'])
y = data['price']

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
print("Validating the model...")
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# Load test data
print("Loading test data...")
test_plate_info_list = get_license_plate_info_list(train=False)

# Convert test data to DataFrame
print("Converting test data to DataFrame...")
test_data = pd.DataFrame([{
    'id': plate_info.id,
    'region': plate_info.region_name,
    'is_government': plate_info.is_government_vehicle,  # 1 for True, 0 for False
    'gov_description': plate_info.government_info['description'],  # Will be label encoded
    'gov_forbidden_to_buy': int(plate_info.government_info['forbidden_to_buy']),  # 1 for True, 0 for False
    'gov_road_advantage': int(plate_info.government_info['road_advantage']),  # 1 for True, 0 for False
    'gov_significance_level': plate_info.government_info['significance_level'],  # Numeric
} for plate_info in test_plate_info_list])

# Ensure 'gov_description' is a string in the test data
test_data['gov_description'] = test_data['gov_description'].astype(str)

# Handle unseen labels in the test set
print("Encoding test data...")

# Get the labels that were seen during training
train_labels = label_encoder.classes_

# Identify the labels in the test data that were not seen during training
test_labels = test_data['gov_description'].unique()

# New labels to handle
unseen_labels = [label for label in test_labels if label not in train_labels]
if unseen_labels:
    print(f"Unseen labels found: {unseen_labels}")
    
    # Manually assign new labels by adding them to the label encoder's classes_
    label_encoder.classes_ = np.append(label_encoder.classes_, unseen_labels)

# Transform the test data's 'gov_description' using the updated encoder
test_data['gov_description'] = label_encoder.transform(test_data['gov_description'])

# One-hot encode the 'region' column in test data
test_data = pd.get_dummies(test_data, columns=['region'], drop_first=True)

# Align test data columns with training data columns
print("Aligning test data columns with training data...")
test_features = test_data.drop(columns=['id'])  # Exclude ID from features
new_test_data = test_features.reindex(columns=X.columns, fill_value=0)

# Predict prices
print("Predicting prices...")
test_data['price'] = model.predict(new_test_data)

# Save predictions to CSV
print("Saving predictions to CSV...")
test_data[['id', 'price']].to_csv('src/data/submissions/ben_fullfeatures_forest_predicted_prices.csv', index=False)
print("Predictions saved.")
