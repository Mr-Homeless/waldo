import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='Split a CSV file into train and test sets.')
parser.add_argument('input_csv', type=str, help='Path to the input CSV file.')
parser.add_argument('output_train', type=str, help='Path to save the train CSV.')
parser.add_argument('output_test', type=str, help='Path to save the test CSV.')
parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the dataset to include in the test split (default: 0.1).')
args = parser.parse_args()
data = pd.read_csv(args.input_csv)
train, test = train_test_split(data, test_size=args.test_size, random_state=42)
train.to_csv(args.output_train, index=False)
test.to_csv(args.output_test, index=False)
print(f"Split {args.input_csv} into {args.output_train} (train) and {args.output_test} (test).")

#python split_csv.py input.csv train.csv test.csv



exit()
'''

# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Explanation:
# - torch: The main PyTorch library
# - nn: Neural Network layers and loss functions
# - optim: Optimization algorithms like SGD, Adam, etc.

# Define a simplified Transformer-based Masked AutoEncoder (MAE)
class SimpleMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMAE, self).__init__()
        
        # Encoder layer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=2),
            num_layers=1
        )
        
        # Decoder layer
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Explanation:
# - Encoder: Transformer layer that compresses the input
# - Decoder: Linear layer to reconstruct the original data



# Define the custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, json_file_path, k=5):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.timestamps = sorted(list(self.data.keys()))
        self.k = k  # Number of sequential timestamps to return

    def __len__(self):
        return len(self.timestamps) - self.k + 1  # Adjust length to accommodate k sequential samples

    def __getitem__(self, idx):
        selected_timestamps = self.timestamps[idx: idx+self.k]
        sequential_data = [self.data[t] for t in selected_timestamps]
        return torch.tensor(sequential_data)

# Explanation:
# - __init__: Reads the JSON and sorts the timestamps
# - __len__: Ensures the dataset length reflects the sequence length (k)
# - __getitem__: Returns k sequential timestamps and their associated data

# Initialize the dataset and data loader
dataset = TimeSeriesDataset(json_file_path='your_data.json', k=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Explanation:
# - Initialize our TimeSeriesDataset
# - DataLoader with batch size of 32 and shuffling enabled


# Model definition
class MaskedAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MaskedAutoEncoder, self).__init__()

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=3
        )
        
        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=3
        )

    # Explanation:
	# - MLP is for feature transformation
	# - Transformer Encoder and Decoder for sequence encoding and decoding
	# - Masking is used before Decoder


    def forward(self, x, mask):
        # MLP layer
        x = self.mlp(x)
        
        # Positional encoding
        # Assuming x shape: [batch_size, seq_len, hidden_dim]
        pos_encoding = torch.arange(0, x.size(1)).unsqueeze(0).float()
        x += pos_encoding
        
        # Add CLS token
        cls_token = torch.zeros(x.size(0), 1, x.size(2))  # Shape: [batch_size, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # New shape: [batch_size, seq_len + 1, hidden_dim]
        
        # Encoder
        x = self.transformer_encoder(x)
        
        # Masking
        masked_x = x * mask.unsqueeze(-1).float()
        
        # Decoder
        output = self.transformer_decoder(masked_x, x)
        
        return output



# Initialize model, optimizer, and loss function
model = MaskedAutoEncoder(input_dim=128, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# Create a mask
for epoch in range(epochs)?:
	for x, y in dataloader:  # Assuming shape: [batch_size, k, input_dim]
	    optimizer.zero_grad()
	    
	    # Forward pass for cheater detection
	    output = model(x)
	    loss = loss_fn(output, y)
	    
	    # Backpropagation
	    loss.backward()
	    optimizer.step()

# Explanation:
# - random_mask is used for 75% masking; adjust accordingly
# - The loss is calculated only for the masked elements

'''