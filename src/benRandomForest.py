import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import get_license_plate_info_list
import numpy as np

# =====================
# Tuned Hyperparameters
# =====================
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
BOOTSTRAP = True
RANDOM_STATE = 42
FILE_NAME = "ben_top_features"

# =====================
# SMAPE Function
# =====================
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# =====================
# Load and Prepare Training Data
# =====================
plate_info_list = get_license_plate_info_list(train=True)

print("lets go!")
data = pd.DataFrame([{
    'id': plate_info.id,
    'region_code': plate_info.region_code,
    'region': plate_info.region_name,
    'plate_digits': plate_info.digits,  
    'is_government': plate_info.is_government_vehicle,
    'road_advantage': plate_info.government_info['road_advantage'],
    'significance_level': plate_info.government_info['significance_level'],
    'price': float(plate_info.price),
} for plate_info in plate_info_list])

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Split into features and target
X = data.drop(columns=['price', 'id'])
y = data['price']

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# =====================
# Train the Tuned Model
# =====================
print("training the tuned model...")
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    bootstrap=BOOTSTRAP,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# =====================
# Evaluate Model
# =====================
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
smape_score = smape(y_val, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"SMAPE: {smape_score:.2f}%")

# Feature Importance (Optional but helpful)
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature importances calculated and sorted:")
print(importance_df.head(15))

# =====================
# Load and Prepare Test Data
# =====================
test_plate_info_list = get_license_plate_info_list(train=False)

test_data = pd.DataFrame([{
    'id': plate_info.id,
    'region_code': plate_info.region_code,
    'region': plate_info.region_name,
    'plate_digits': plate_info.digits,
    'is_government': plate_info.is_government_vehicle,
    'road_advantage': plate_info.government_info['road_advantage'],
    'significance_level': plate_info.government_info['significance_level'],
} for plate_info in test_plate_info_list])

test_data = pd.get_dummies(test_data, columns=['region'], drop_first=True)

# Align test features with training columns
test_features = test_data.drop(columns=['id'])
test_features = test_features.reindex(columns=X.columns, fill_value=0)

# =====================
# Predict and Save
# =====================
test_data['price'] = model.predict(test_features)
test_data[['id', 'price']].to_csv(f'src/data/submissions/{FILE_NAME}.csv', index=False)
