Tried all features (except for date) full random forest: 98.9

Grid search 60.14 
Best hyperparameters: {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Best model selected based on grid search.
Feature importances calculated and sorted:
                             Feature    Importance
4                       plate_digits  5.741917e-01
3                        region_code  1.847720e-01
2                 significance_level  8.896989e-02
40                     region_Moscow  2.398505e-02
26            region_Kemerovo Oblast  2.126145e-02
20   region_Jewish Autonomous Oblast  2.093736e-06
17  region_Donetsk People's Republic  1.682789e-06
38  region_Luhansk People's Republic  9.086658e-07
43    region_Nenets Autonomous Okrug  3.472056e-07
10                   region_Baikonur  0.000000e+00

SMAPE: 96.18%
N_ESTIMATORS = 150
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42

SMAPE: 58.43%
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
BOOTSTRAP = True
RANDOM_STATE = 42
'id': plate_info.id,
'region_code': plate_info.region_code, 
'region': plate_info.region_name,
'plate_digits': plate_info.digits, 
'is_government': plate_info.is_government_vehicle,
'road_advantage': plate_info.government_info['road_advantage'],
'significance_level': plate_info.government_info['significance_level'],