import pandas as pd

train_data = pd.read_csv('./train.csv', dtype=dtype_dict)
print(train_data.head())