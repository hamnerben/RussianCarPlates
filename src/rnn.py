import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from parse_plate import *
import csv
import utils
import re

# Example data
prices = []
region_names = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Open the train CSV file
plate_info_list = utils.get_license_plate_info_list(train=True)

# Populate region_names with valid region names from plate_info_list
for plate_info in plate_info_list:
    if plate_info.valid:
        region_names.append(plate_info.region_name)
        prices.append(float(plate_info.price))  # Assuming price is a float



# Step 1: Tokenize the region names
def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

vocab = {}

def build_vocab(region_names):
    for name in region_names:
        for word in tokenizer(name):
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Reserve 0 for padding
    return vocab

vocab = build_vocab(region_names)

# Convert region names to sequences of integers
def encode(region_name):
    return [vocab.get(word, 0) for word in tokenizer(region_name)]

X = [encode(name) for name in region_names]

# Step 2: Pad sequences to ensure they have the same length
max_length = max([len(seq) for seq in X])  # maximum sequence length
X_padded = [seq + [0] * (max_length - len(seq)) for seq in X]

# Step 3: Convert to tensors
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(prices, dtype=torch.float32)

X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
# Step 4: Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        out, _ = self.rnn(x)  # RNN output
        out = out[:, -1, :]  # Get the last hidden state
        out = self.fc(out)  # Pass through the fully connected layer
        return out

# Initialize model, loss function, and optimizer
vocab_size = len(vocab) + 1  # Plus one for padding token
embed_size = 50
hidden_size = 64
model = RNNModel(vocab_size, embed_size, hidden_size)
#use GPU if available
model.to(device)
# Move tensors to the same device
# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        
        # Calculate loss
        loss = criterion(outputs.view(-1), y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print("epoch", epoch)
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Step 7: Make a prediction
model.eval()
test_region = []
test_id = [] 
test_plates = utils.get_license_plate_info_list(train=False)
for plate_info in test_plates:
    if plate_info.valid:
        test_region.append(plate_info.region_name)
        test_id.append(plate_info.id)
test_seq_padded = []
for region_name in test_region:
    test_seq = encode(region_name)
    padded_seq = test_seq + [0] * (max_length - len(test_seq))
    test_seq_padded.append(padded_seq)

test_seq_padded = torch.tensor(test_seq_padded, dtype=torch.long)
test_seq_padded = test_seq_padded.to(device)
model.to(device)

with torch.no_grad():
    predicted_prices = model(test_seq_padded).squeeze().tolist()  # Convert predictions to a list
    # Export predictions to a CSV file
    output_file = '/home/price373/CS5665/RussianPlates/RussianCarPlates/predictions.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Price'])  # Write header

        for idx, region_name in enumerate(test_region):
            test_seq = encode(region_name)
            test_seq_padded = torch.tensor([test_seq + [0] * (max_length - len(test_seq))], dtype=torch.long).to(device)
            
            with torch.no_grad():
                predicted_price = model(test_seq_padded).item()
                print(f"Predicted price for {region_name}: {predicted_price}")
            predicted_price = predicted_prices[idx]
            writer.writerow([test_id[idx], predicted_price])  # Write ID and predicted price

    print(f"Predictions exported to {output_file}")