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
from sklearn.preprocessing import OneHotEncoder

possible_letters = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']
# Example data
prices = []
region_names = []
is_government = []
letters = []
numbers = []
region_code = []
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Open the train CSV file
plate_info_list = utils.get_license_plate_info_list(train=True)

# Populate region_names with valid region names from plate_info_list
for plate_info in plate_info_list:
    if plate_info.valid:
        region_names.append(plate_info.region_name)
        prices.append(float(plate_info.price))  # Assuming price is a float
        if plate_info._is_government_vehicle:
            is_government.append(1)
        else:
            is_government.append(0)
        # Convert letters to their corresponding indices
        liscense_letters = plate_info.letter1 + plate_info.letter2 + plate_info.letter3
        # print(liscense_letters)
        letters.append(liscense_letters)
        numbers.append(plate_info.digits)
        region_code.append(plate_info.region_code)

    # Step 1: Tokenize the region names and letters
def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

vocab = {}

def build_vocab(region_names, letters):
    for name in region_names:
        for word in tokenizer(name):
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Reserve 0 for padding
    for letter_seq in letters:
        for letter in letter_seq:
            if letter not in vocab:
                vocab[letter] = len(vocab) + 1  # Add letters to vocab
    return vocab

vocab = build_vocab(region_names, letters)

# Convert region names and letters to sequences of integers
def encode(text):
    return [vocab.get(word, 0) for word in tokenizer(text)]

def encode_letters(letter_seq):
    return [vocab.get(letter, 0) for letter in letter_seq]

X = [encode(name) for name in region_names]

# Step 2: Pad sequences to ensure they have the same length
max_length = max([len(seq) for seq in X])  # maximum sequence length
X_padded = [seq + [0] * (max_length - len(seq)) for seq in X]

letters_encoded = [encode_letters(letter_seq) for letter_seq in letters]
max_letters_length = max([len(seq) for seq in letters_encoded])
letters_padded = [seq + [0] * (max_letters_length - len(seq)) for seq in letters_encoded]

# Step 3: One-hot encode the is_government, numbers, and region_code features
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# One-hot encode region codes
region_code_reshaped = np.array(region_code).reshape(-1, 1)
region_code_one_hot = one_hot_encoder.fit_transform(region_code_reshaped)

# One-hot encode is_government
is_government_reshaped = np.array(is_government).reshape(-1, 1)
is_government_one_hot = one_hot_encoder.fit_transform(is_government_reshaped)

# One-hot encode numbers
numbers_reshaped = np.array(numbers).reshape(-1, 1)
numbers_one_hot = one_hot_encoder.fit_transform(numbers_reshaped)

# Combine all features
X_with_features = []
for seq, letter_seq, num_one_hot, code_one_hot, gov_one_hot in zip(
    X_padded, letters_padded, numbers_one_hot, region_code_one_hot, is_government_one_hot
):
    combined_features = seq + letter_seq + num_one_hot.tolist() + code_one_hot.tolist() + gov_one_hot.tolist()
    X_with_features.append(combined_features)

# Step 4: Convert to tensors
X_tensor = torch.tensor(X_with_features, dtype=torch.long)
y_tensor = torch.tensor(prices, dtype=torch.float32)

X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)

# Step 5: Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.3):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + hidden_size +  # For region names and letters
                            len(numbers_one_hot[0]) + len(region_code_one_hot[0]) + len(is_government_one_hot[0]), 1)

    def forward(self, x, mask=None):
        """
        x: Tensor of shape (batch_size, seq_len)
        mask: Optional Tensor of shape (batch_size, seq_len), 1 for real tokens, 0 for padding
        """
        # Split input into different feature groups
        seq_len = x.size(1)
        region_seq_len = max_length
        letters_seq_len = max_letters_length

        region_x = x[:, :region_seq_len]
        letters_x = x[:, region_seq_len:region_seq_len + letters_seq_len]
        other_features = x[:, region_seq_len + letters_seq_len:]

        # Process region names
        region_embedded = self.embedding(region_x)  # (batch_size, region_seq_len, embed_size)
        region_lstm_out, _ = self.lstm(region_embedded)  # (batch_size, region_seq_len, hidden_size)

        # Process letters
        letters_embedded = self.embedding(letters_x)  # (batch_size, letters_seq_len, embed_size)
        letters_lstm_out, _ = self.lstm(letters_embedded)  # (batch_size, letters_seq_len, hidden_size)

        # Apply mask to region names
        if mask is not None:
            region_mask = mask[:, :region_seq_len].unsqueeze(-1)  # (batch_size, region_seq_len, 1)
            region_lstm_out = region_lstm_out * region_mask
            region_sum_hidden = region_lstm_out.sum(dim=1)
            region_lengths = region_mask.sum(dim=1).clamp(min=1e-5)  # avoid division by zero
            region_pooled = region_sum_hidden / region_lengths
        else:
            region_pooled = region_lstm_out.mean(dim=1)

        # Apply mask to letters
        if mask is not None:
            letters_mask = mask[:, region_seq_len:region_seq_len + letters_seq_len].unsqueeze(-1)  # (batch_size, letters_seq_len, 1)
            letters_lstm_out = letters_lstm_out * letters_mask
            letters_sum_hidden = letters_lstm_out.sum(dim=1)
            letters_lengths = letters_mask.sum(dim=1).clamp(min=1e-5)  # avoid division by zero
            letters_pooled = letters_sum_hidden / letters_lengths
        else:
            letters_pooled = letters_lstm_out.mean(dim=1)

        # Concatenate all features
        combined_features = torch.cat([region_pooled, letters_pooled, other_features.float()], dim=1)

        # Fully connected layer
        out = self.dropout(combined_features)
        price_pred = self.fc(out)
        return price_pred

# Initialize model
vocab_size = len(vocab) + 1  # Plus one for padding token
embed_size = 50
hidden_size = 64
num_layers = 2  # Specify the number of LSTM layers
dropout = 0.3  # Specify the dropout rate
model = RNNModel(vocab_size, embed_size, hidden_size, num_layers=num_layers, dropout=dropout)
#use GPU if available
model.to(device)
# Move tensors to the same device
# Define loss function and optimizer
criterion = nn.HuberLoss(delta=1.0)  # Huber loss is robust to outliers
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
epochs = 1
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        
        # Create a mask for padding (1 for real tokens, 0 for padding)
        mask = (X_batch != 0).float()
        
        # Forward pass
        outputs = model(X_batch, mask=mask)
        
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
    # Save the model
    torch.save(model.state_dict(), 'rnn_model.pth')

# Step 7: Make a prediction
model.eval()
test_region_names = []
test_id = []
test_is_government = []
test_letters = []
test_numbers = []
test_region_code = []

test_plates = utils.get_license_plate_info_list(train=False)
for plate_info in test_plates:
    if plate_info.valid:
        test_region_names.append(plate_info.region_name)
        test_id.append(plate_info.id)
        if plate_info._is_government_vehicle:
            test_is_government.append(1)
        else:
            test_is_government.append(0)
        
        liscense_letters = plate_info.letter1 + plate_info.letter2 + plate_info.letter3
        test_letters.append(liscense_letters)
        test_numbers.append(plate_info.digits)
        test_region_code.append(plate_info.region_code)

# Encode and pad region names
# Create a new OneHotEncoder for the validation set
test_X = [encode(name) for name in test_region_names]

# Step 2: Pad sequences to ensure they have the same length
test_max_length = max([len(seq) for seq in X])  # maximum sequence length
test_X_padded = [seq + [0] * (test_max_length - len(seq)) for seq in test_X]

test_letters_encoded = [encode_letters(letter_seq) for letter_seq in test_letters]
test_max_letters_length = max([len(seq) for seq in test_letters_encoded])
test_letters_padded = [seq + [0] * (test_max_letters_length - len(seq)) for seq in test_letters_encoded]

# Step 3: One-hot encode the is_government, numbers, and region_code features
# Use the already fitted one-hot encoder from training
test_region_code_reshaped = np.array(test_region_code).reshape(-1, 1)
test_region_code_one_hot = one_hot_encoder.transform(test_region_code_reshaped)

test_is_government_tensor = torch.tensor(test_is_government, dtype=torch.float32).unsqueeze(1)
test_is_government_one_hot = test_is_government_tensor.numpy().tolist()  # Convert to list for concatenation

test_numbers_reshaped = np.array(test_numbers).reshape(-1, 1)
test_numbers_one_hot = one_hot_encoder.transform(test_numbers_reshaped)

# Combine all features
test_X_with_features = []
for seq, letter_seq, num_one_hot, code_one_hot, gov_one_hot in zip(
    test_X_padded, test_letters_padded, test_numbers_one_hot, test_region_code_one_hot, test_is_government_one_hot
):
    test_combined_features = seq + letter_seq + num_one_hot.tolist() + code_one_hot.tolist() + gov_one_hot
    test_X_with_features.append(test_combined_features)

# Step 4: Convert to tensors
test_X_tensor = torch.tensor(test_X_with_features, dtype=torch.long)
test_X_tensor = test_X_tensor.to(device)

# Create a DataLoader for the validation set
test_dataset = TensorDataset(test_X_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Load the trained model
model.to(device)

# Make predictions
predicted_prices = []
model.eval()
with torch.no_grad():
    for i in range(len(test_X_tensor)):
        X_batch = test_X_tensor[i].unsqueeze(0)  # Add batch dimension
        output = model(X_batch).squeeze()
        predicted_prices.append(output.item())

# Export predictions to a CSV file
output_file = '/home/price373/CS5665/RussianPlates/RussianCarPlates/predictions.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Price'])  # Write header

    for idx, predicted_price in enumerate(predicted_prices):
        writer.writerow([test_id[idx], predicted_price])

print(f"Predictions exported to {output_file}")
