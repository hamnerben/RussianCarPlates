import matplotlib.pyplot as plt

# for forest
timestamps = ['Before tuning', 'Intitial GridSearch', 'Optimized Parameters', 'Added Year']
smape_values = [98.9426, 70.0413, 60.1560, 58.500]

# for lstm
epochsName = ['LSTM1Epoch', 'LSTM5Epoch', '10Epoch', 'LSTM15Epoch', 'LSTM50Epoch', 'LSTM75Epoch', 'LSTM_Huber', 'LSTM_MSE']
score = [194.46, 174.86, 154.49, 137.4, 78, 81, 77, 78.9]

# Plot SMAPE
plt.figure(figsize=(10, 5))
plt.bar(epochsName, score, color='lightblue')
# plt.bar(timestamps, smape_values, color='lightgreen')

plt.title('Impact of Epochs Tuning on SMAPE Score')
# plt.title('Impact of Hyperparameter Tuning on SMAPE Score')
plt.ylabel('SMAPE (%)')
plt.gca().set_ylim(bottom=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the graph to a file
plt.savefig('docs/lstm_epoch_tuning.png', format='png', dpi=300)
# plt.savefig('docs/smape_tuning.png', format='png', dpi=300)

# Show the graph
plt.show()