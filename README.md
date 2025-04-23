basic summary: predicting the price of Russian plates
- each one is 6 characters (3 letters, 3 numbers) and then a region/government code
- submission SMAPE scores closer to 0 are better
- predicting a higher cost than the actual is better than under predicting

we ended up using random forest and LSTM models, each was run with minimal features to get a baseline.
