import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import utils
import csv
import re
from datetime import datetime
import time
import pandas as pd
import seaborn as sns


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
# Split dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# --- Model ---
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
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
# Hyperparameters to run to find the best one
# parameters[0] = learning rate, [1] = hidden size, [2] = dropout

features = ['learning_rate', 'hidden_size', 'dropout']
feature = 0 ## i in the loop
parameters = [[0.5], [2, 4, 8, 16], [0.3,0.5,0.1,0]]
parameter = 0 ## j in the loop
vocab_size = len(vocab) + 1
embed_size = 50
num_layers = 4
smape_scores = [[],[],[]]  
for feature in range(len(features)):
    for parameter in range(len(parameters[feature])):

        if feature == 0:
            lr = parameters[feature][parameter]
            epochs = 1
        else:
            lr = 0.5
        
        if feature == 1:
            hidden_size = parameters[feature][parameter]
        else:
            hidden_size = 2
        if feature == 2:
            dropout = parameters[feature][parameter]
        else:
            dropout = 0.3
        print(f"Training with {features[feature]} = {parameters[feature][parameter]}")
            
        model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # tracking for plots
        train_losses = []
        val_losses = []
        # --- Training ---  
        epochs = 25
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_dataloader:
                X_batch, y_batch = batch
                optimizer.zero_grad()
                mask = (X_batch != 0).float()
                preds = model(X_batch, mask=mask)
                loss = criterion(preds.view(-1), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_dataloader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            valTotal_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    X_batch, y_batch = batch
                    mask = (X_batch != 0).float()
                    preds = model(X_batch, mask=mask)
                    loss = criterion(preds.view(-1), y_batch)
                    valTotal_loss += loss.item()
            val_loss = valTotal_loss / len(val_dataloader)
            val_losses.append(val_loss)    
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            # Calculate SMAPE for validation predictions
        smape_total = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, y_batch = batch
                mask = (X_batch != 0).float()
                preds = model(X_batch, mask=mask)
                smape = 100 * torch.mean(
                    2 * torch.abs(preds.view(-1) - y_batch) / (torch.abs(preds.view(-1)) + torch.abs(y_batch) + 1e-5)
                )
                smape_total += smape.item()
        smape_val = smape_total / len(val_dataloader)
        print(f"SMAPE: {smape_val:.2f}%")
        smape_scores[feature].append(smape_val)

        torch.save(model.state_dict(), f"lstm_model_HP_{features[feature]}-{parameters[feature][parameter]}.pth")
        plt.clf()  # Clear the current figure to avoid overlapping
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"loss_plot_HP_{features[feature]}-{parameters[feature][parameter]}.png")  # Save the figure to a file

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
        filename = f"lstm_HP_{features[feature]}-{parameters[feature][parameter]}"
        # --- Save to CSV ---
        with open(f'src/data/submissions/{filename}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'price'])
            writer.writerows(zip(test_id, predictions))

        print(f"✅ Predictions exported to {filename}.csv")

    # --- Table of SMAPE scores ---
print("\nSMAPE Scores Table:")
print(f"{'Feature':<15}{'Parameter':<15}{'SMAPE Score':<15}")
print("-" * 45)
for feature_idx, feature_name in enumerate(features):
    for param_idx, param_value in enumerate(parameters[feature_idx]):
        smape_score = smape_scores[feature_idx][param_idx]
        # --- Export SMAPE Scores Table as PNG ---

        # Create a DataFrame for SMAPE scores
        smape_data = []
        for feature_idx, feature_name in enumerate(features):
            for param_idx, param_value in enumerate(parameters[feature_idx]):
                smape_score = smape_scores[feature_idx][param_idx]
                smape_data.append([feature_name, param_value, smape_score])

        smape_df = pd.DataFrame(smape_data, columns=["Feature", "Parameter", "SMAPE Score"])

        # Create a table and save it as a PNG file
        plt.figure(figsize=(10, 6))
        plt.axis('off')  # Turn off the axes
        table = plt.table(cellText=smape_df.values, colLabels=smape_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(smape_df.columns))))  # Adjust column widths

        # Save the table as a PNG file
plt.savefig("smape_scores_table.png", bbox_inches='tight', dpi=300)
print("✅ SMAPE Scores Table exported as smape_scores_table.png")