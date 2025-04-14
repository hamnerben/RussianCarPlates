import matplotlib.pyplot as plt


timestamps = ['Before tuning', 'Intitial GridSearch', 'Optimized Parameters', 'Added Year']
smape_values = [98.9426, 70.0413, 60.1560, 58.500]

epochsName = [ 'LSTM5Epoch', '10Epoch', 'LSTM15Epoch', 'LSTM50Epoch', 'LSTM75Epoch', 'LSTM_Huber', 'LSTM_MSE', 'LSTM1Epoch']
score = ['174.86', '154.49', '137.4', '78', '81', '77', '78.9', '194.46']

# Plot SMAPE
plt.figure(figsize=(10, 5))
plt.bar(epochsName, score, color='lightblue')
plt.title('Impact of Epochs Tuning on SMAPE Score')
plt.ylabel('SMAPE (%)')
plt.gca().set_ylim(bottom=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the graph to a file
plt.savefig('docs/lstm_epoch_tuning.png', format='png', dpi=300)

# Show the graph
plt.show()