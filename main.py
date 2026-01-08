from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

# Loading dataset
data = pd.read_csv('C:/Users/sives/Downloads/sampled_data.csv')

# Define the features and targe variable
X = data.drop(columns=['Label', 'ClassLabel'])  # Adjust these columns as needed
y = data['Label']  # Use the appropriate label column

# Checking class distribution
print("Class distribution before resampling:", Counter(y))

# Define samplig strategy
sampling_strategy_smote = {}
sampling_strategy_random = {}
for label, count in Counter(y).items():
    if count == 1:
        sampling_strategy_random[label] = 50
    elif count < 50000:
        sampling_strategy_smote[label] = 50000
    else:
        sampling_strategy_smote[label] = count

# Apply RandomOerSampler and SMOTE
ros = RandomOverSampler(sampling_strategy=sampling_strategy_random, random_state=42)
X_res, y_res = ros.fit_resample(X, y)

smote = SMOTE(sampling_strategy=sampling_strategy_smote, k_neighbors=1, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_res, y_res)
print("Class distribution after resampling:", Counter(y_train_res))

# Encode label and scaled features
label_encoder = LabelEncoder()
y_train_res_encoded = label_encoder.fit_transform(y_train_res)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add channel dimension for CNN
y_train_tensor = torch.tensor(y_train_res_encoded, dtype=torch.long)

# Create the  DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Define the Lightweiht CNN-LSTM Model
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calculate cnn_output_size based on the input and pooling layers
        cnn_output_size = input_size // 4
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Resape for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# Instantiate the model, loss function, and optimizer
input_size = X_train_tensor.shape[2]  # Input size afte scaling and adding channel dimension
hidden_size = 128
num_classes = len(label_encoder.classes_)
model = LightweightCNNLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculating accuracy
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = correct / total * 100
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

print("Training completed.")
