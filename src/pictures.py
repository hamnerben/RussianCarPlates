import matplotlib.pyplot as plt


timestamps = ['Before tuning', 'Intitial GridSearch', 'Optimized Parameters', 'Added Year']
smape_values = [98.9426, 70.0413, 60.1560, 58.500]

# Plot SMAPE
plt.figure(figsize=(10, 5))
plt.bar(timestamps, smape_values, color='lightgreen')
plt.title('Impact of Hyperparameter Tuning on SMAPE score')
plt.ylabel('SMAPE (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the graph to a file
plt.savefig('docs/smape_tuning.png', format='png', dpi=300)

# Show the graph
plt.show()