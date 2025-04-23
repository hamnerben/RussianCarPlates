import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import utils
import csv
from datetime import datetime

filename = "ben_rnn_with_year"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA:", torch.cuda.is_available())

# Prepare training data
prices, is_government, letters, numbers, region_code, years = [], [], [], [], [], []

train_plates = utils.get_license_plate_info_list(train=True)
for plate in train_plates:
    if plate.valid:
        prices.append(float(plate.price))
        is_government.append('1' if plate._is_government_vehicle else '0')
        letters.append(plate.letter1 + plate.letter2 + plate.letter3)
        numbers.append(plate.digits)
        region_code.append(plate.region_code)
        years.append(str(datetime.strptime(plate.date, '%Y-%m-%d %H:%M:%S').year))  # Extract year

# --- Build vocabulary for letters ---
vocab = {}
def build_vocab(letters):
    for seq in letters:
        for letter in seq:
            if letter not in vocab:
                vocab[letter] = len(vocab) + 1
    return vocab

vocab = build_vocab(letters)

def encode_letters(seq):
    return [vocab.get(l, 0) for l in seq]

letters_encoded = [encode_letters(seq) for seq in letters]
max_letters_length = max(len(seq) for seq in letters_encoded)
letters_padded = [seq + [0] * (max_letters_length - len(seq)) for seq in letters_encoded]

# --- One-hot encoders (fit only on training data) ---
region_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
gov_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
number_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
year_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

region_code_one_hot = region_encoder.fit_transform(np.array(region_code).reshape(-1, 1))
is_government_one_hot = gov_encoder.fit_transform(np.array(is_government).reshape(-1, 1))
numbers_one_hot = number_encoder.fit_transform(np.array(numbers).reshape(-1, 1))
years_one_hot = year_encoder.fit_transform(np.array(years).reshape(-1, 1))

# Combine all features
X_with_features = []
for l_seq, n_onehot, r_onehot, g_onehot, y_onehot in zip(
    letters_padded, numbers_one_hot, region_code_one_hot, is_government_one_hot, years_one_hot
):
    combined = l_seq + n_onehot.tolist() + r_onehot.tolist() + g_onehot.tolist() + y_onehot.tolist()
    X_with_features.append(combined)

X_tensor = torch.tensor(X_with_features, dtype=torch.long).to(device)
y_tensor = torch.tensor(prices, dtype=torch.float32).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Model ---
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.3):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        self.fc_input_size = hidden_size + \
                             region_code_one_hot.shape[1] + \
                             is_government_one_hot.shape[1] + \
                             numbers_one_hot.shape[1] + \
                             years_one_hot.shape[1]
        self.fc = nn.Linear(self.fc_input_size, 1)

    def forward(self, x, mask=None):
        letters_seq_len = max_letters_length
        letters_x = x[:, :letters_seq_len]
        other_features = x[:, letters_seq_len:]

        letters_emb = self.embedding(letters_x)
        lstm_out, _ = self.lstm(letters_emb)

        if mask is not None:
            letters_mask = mask[:, :letters_seq_len].unsqueeze(-1)
            lstm_out = lstm_out * letters_mask
            pooled = lstm_out.sum(dim=1) / letters_mask.sum(dim=1).clamp(min=1e-5)
        else:
            pooled = lstm_out.mean(dim=1)

        combined = torch.cat([pooled, other_features.float()], dim=1)
        out = self.dropout(combined)
        return self.fc(out)

# --- Init model ---
vocab_size = len(vocab) + 1
embed_size = 50
hidden_size = 64
num_layers = 2
dropout = 0.3

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)

criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
epochs = 1
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        mask = (X_batch != 0).float()
        preds = model(X_batch, mask=mask)
        loss = criterion(preds.view(-1), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), "rnn_model.pth")

# --- Prepare test data ---
test_id, test_letters, test_numbers, test_region_code, test_is_gov, test_years = [], [], [], [], [], []

test_plates = utils.get_license_plate_info_list(train=False)
for plate in test_plates:
    if plate.valid:
        test_id.append(plate.id)
        test_letters.append(plate.letter1 + plate.letter2 + plate.letter3)
        test_numbers.append(plate.digits)
        test_region_code.append(plate.region_code)
        test_is_gov.append('1' if plate._is_government_vehicle else '0')
        test_years.append(str(datetime.strptime(plate.date, '%Y-%m-%d %H:%M:%S').year))  # Include year

test_letters_encoded = [encode_letters(seq) for seq in test_letters]
test_letters_padded = [seq + [0] * (max_letters_length - len(seq)) for seq in test_letters_encoded]

test_region_code_one_hot = region_encoder.transform(np.array(test_region_code).reshape(-1, 1))
test_is_government_one_hot = gov_encoder.transform(np.array(test_is_gov).reshape(-1, 1))
test_numbers_one_hot = number_encoder.transform(np.array(test_numbers).reshape(-1, 1))
test_years_one_hot = year_encoder.transform(np.array(test_years).reshape(-1, 1))

test_X_with_features = []
for l_seq, n_onehot, r_onehot, g_onehot, y_onehot in zip(
    test_letters_padded, test_numbers_one_hot, test_region_code_one_hot, test_is_government_one_hot, test_years_one_hot
):
    combined = l_seq + n_onehot.tolist() + r_onehot.tolist() + g_onehot.tolist() + y_onehot.tolist()
    test_X_with_features.append(combined)

test_X_tensor = torch.tensor(test_X_with_features, dtype=torch.long).to(device)

# --- Predict ---
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_X_tensor:
        batch = batch.unsqueeze(0)
        mask = (batch != 0).float()
        pred = model(batch, mask=mask)
        predictions.append(pred.item())

# --- Save to CSV ---
with open(f'src/data/submissions/{filename}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'price'])
    writer.writerows(zip(test_id, predictions))

print(f"✅ Predictions exported to {filename}.csv")
