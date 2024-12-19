import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv

# Define the LSTM network
class LSTMNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=3, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Output for the last time step
        return out

# Define custom Dataset
class LSTMCustomDataset(Dataset):
    def __init__(self, input_file, ground_truth_file, time_steps=10, augment_r_encode=None, augment_l_encode=None):
        self.inputs = pd.read_csv(input_file)
        self.ground_truth = pd.read_csv(ground_truth_file)
        
        # Match inputs and ground truth by frame number
        self.data = self.inputs.merge(self.ground_truth, left_on='frame_number', right_on='Frame')
        self.data = self.data.drop(columns=['Frame'])

        # Apply augmentation
        self.augmented_data = []
        r_augment = augment_r_encode or [0]
        l_augment = augment_l_encode or [0]

        for r_offset in r_augment:
            for l_offset in l_augment:
                temp_data = self.data.copy()
                temp_data['r_encode'] += r_offset
                temp_data['l_encode'] += l_offset

                # Apply normalization after augmentation
                temp_data['bottom_left_x'] /= 660
                temp_data['bottom_left_y'] /= 616
                temp_data['box_angle'] /= 90
                temp_data['r_encode'] /= 90
                temp_data['l_encode'] /= 100

                self.augmented_data.append(temp_data)

        # Combine all augmented data
        self.data = pd.concat(self.augmented_data, ignore_index=True)

        # Normalize target values
        self.data['x_tip'] /= 660  # Normalize x_tip
        self.data['y_tip'] /= 616  # Normalize y_tip
        self.data['angle'] /= 90   # Normalize angle

        # Create sequences of 10 time steps
        self.time_steps = time_steps
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        input_features = self.data[['bottom_left_x', 'bottom_left_y', 'box_angle', 'r_encode', 'l_encode']].values
        labels = self.data[['x_tip', 'y_tip', 'angle']].values
        frame_numbers = self.data['frame_number'].values

        for i in range(len(input_features) - self.time_steps + 1):
            # Ensure the frames are continuous for time step
            if np.all(frame_numbers[i:i+self.time_steps] == np.arange(frame_numbers[i], frame_numbers[i] + self.time_steps)):
                input_seq = input_features[i:i+self.time_steps]
                target = labels[i + self.time_steps - 1]  # Target is the last frame's ground truth
                sequences.append((input_seq, target))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_input, target = self.sequences[idx]
        return torch.tensor(seq_input, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, log_file='training_log.csv'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Logging to CSV
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                # LSTM requires input as (batch, time_step, input_size)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
                optimizer.step()
                train_loss += loss.item()
            
            # Validation step
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Write losses to the log file
            writer.writerow([epoch+1, train_loss, val_loss])
                

# Main
if __name__ == "__main__":
    # Augmentation values
    r_encode_augment = [-2, -1, 0, 1, 2]
    l_encode_augment = [-1, 0, 1]

    # Load datasets
    train_dataset = LSTMCustomDataset('lstm_inputs_train.csv', 'ground_truth_train.csv', time_steps=10, augment_r_encode=r_encode_augment, augment_l_encode=l_encode_augment)
    val_dataset = LSTMCustomDataset('lstm_inputs_val.csv', 'ground_truth_val.csv', time_steps=10, augment_r_encode=r_encode_augment, augment_l_encode=l_encode_augment)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = LSTMNetwork()

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0005, log_file='training_log.csv')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model training completed and saved as 'lstm_model.pth'.")
