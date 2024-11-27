#
#


# # main.py
# from federated_training import start_federated_training
# from model import create_FA3D_model
#
# def main():
#     # Start federated learning training
#     start_federated_training()
#
# if __name__ == "__main__":
#     main()


from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
#
#
# # Function to remove rare classes
# def remove_rare_classes(X, y, min_samples=2):
#     # Count instances of each class
#     class_counts = pd.Series(y).value_counts()
#
#     # Find classes with fewer than the minimum number of samples
#     rare_classes = class_counts[class_counts < min_samples].index
#
#     # Filter out rare classes
#     filter_mask = ~np.isin(y, rare_classes)
#     X_filtered = X[filter_mask]
#     y_filtered = y[filter_mask]
#
#     print(f"Classes removed: {rare_classes}")
#     return X_filtered, y_filtered
#
#
# # Function to preprocess data with stratified sampling and smaller batch size
# def preprocess_data_in_batches(X, y, batch_size=1000):  # Reduced batch size
#     # Remove rare classes
#     X, y = remove_rare_classes(X, y)
#
#     # Initialize RandomOverSampler and SMOTE
#     ros = RandomOverSampler(random_state=42)
#     smote = SMOTE(random_state=42, k_neighbors=1)
#
#     # Check overall class distribution after removal
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     print("Class distribution before resampling:", dict(zip(unique_classes, class_counts)))
#
#     # Split data into batches using StratifiedShuffleSplit
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=batch_size, random_state=42)  # Increased splits
#
#     batch_num = 0
#     X_res = []
#     y_res = []
#
#     for train_index, test_index in sss.split(X, y):
#         # Get the batch data
#         X_batch, y_batch = X[train_index], y[train_index]
#
#         # Check if the batch contains more than one class
#         if len(np.unique(y_batch)) > 1:
#             # Apply RandomOverSampler to handle minority classes
#             X_res_batch, y_res_batch = ros.fit_resample(X_batch, y_batch)
#
#             # Apply SMOTE to handle class imbalance
#             X_res_batch, y_res_batch = smote.fit_resample(X_res_batch, y_res_batch)
#
#             # Append to the final lists
#             X_res.append(X_res_batch)
#             y_res.append(y_res_batch)
#             batch_num += 1
#         else:
#             print(f"Skipping batch {batch_num + 1} because it contains only one class.")
#             # Fallback: Re-sample or copy data points to make the batch usable
#             if batch_num == 0:  # Example logic for fallback if no usable batch
#                 X_res.append(X_batch)
#                 y_res.append(y_batch)
#                 batch_num += 1
#
#     # Combine the resampled data from all batches
#     if X_res and y_res:
#         X_res = np.vstack(X_res)
#         y_res = np.hstack(y_res)
#     else:
#         X_res = np.empty((0, X.shape[1]))
#         y_res = np.empty(0)
#
#     print(f"Total usable batches: {batch_num}")
#     return X_res, y_res
#
#
# # Load dataset
# data = pd.read_csv('C:/Users/sives/Downloads/sampled_data.csv')
#
# # Define features and target variable
# X = data.drop(columns=['Label', 'ClassLabel'])  # Adjust these columns as needed
# y = data['Label']  # Use the appropriate label column
#
# # Preprocess data with batch resampling
# X_res, y_res = preprocess_data_in_batches(X.values, y.values)
#
# # Encode the labels to integers
# from sklearn.preprocessing import LabelEncoder
#
# label_encoder = LabelEncoder()
# y_res_encoded = label_encoder.fit_transform(y_res)
#
# # Normalize features using StandardScaler
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_res)
#
# # Convert data to PyTorch tensors
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
# X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
# y_tensor = torch.tensor(y_res_encoded, dtype=torch.long)  # Now encoded labels
#
# # Create DataLoader for batching
# train_dataset = TensorDataset(X_tensor, y_tensor)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # Print a sample of the data
# print(f"Sample data shape: {X_tensor.shape}, {y_tensor.shape}")
# import time
# import pandas as pd
# from sklearn.model_selection import StratifiedShuffleSplit
#
#
# # Function to preprocess data in batches
# def preprocess_data_in_batches(X, y, n_splits=10):
#     start_time = time.time()  # Start timing
#
#     # Check initial dataset size
#     print(f"Initial shape of X: {X.shape}")
#     print(f"Initial shape of y: {y.shape}")
#
#     # Create Stratified Shuffle Split for resampling
#     sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
#
#     # Initialize list to store batches
#     batches = []
#
#     try:
#         # Loop over the splits
#         for batch_num, (train_index, test_index) in enumerate(sss.split(X, y)):
#             print(f"Processing batch {batch_num + 1}/{n_splits}")
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#             print(f"Batch {batch_num + 1} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#             print(f"Batch {batch_num + 1} - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#
#             batches.append((X_train, X_test, y_train, y_test))
#
#         print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
#         return batches
#
#     except Exception as e:
#         print(f"Error during preprocessing: {e}")
#         return None
#
#
# # Sample DataFrame for testing
# df = pd.read_csv("C:/Users/sives/Downloads/cic-collection.csv")  # Update this with your dataset path
#
# # Features and target
# X = df.drop(columns=["ClassLabel"])
# y = df["ClassLabel"]
#
# # Start data preprocessing
# print("Starting data preprocessing")
# batches = preprocess_data_in_batches(X, y, n_splits=5)  # Example with 5 splits for debugging
#
# # Final Output
# if batches:
#     print(f"Total usable batches: {len(batches)}")
# else:
#     print("No batches generated.")

# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from collections import Counter
#
# # Load dataset
# data = pd.read_csv('C:/Users/sives/Downloads/sampled_data.csv')
#
# # Define features and target variable
# X = data.drop(columns=['Label', 'ClassLabel'])  # Adjust these columns as needed
# y = data['Label']  # Use the appropriate label column
#
# # Check class distribution
# print("Class distribution before resampling:", Counter(y))
#
# # Define sampling strategy
# sampling_strategy_smote = {}
# sampling_strategy_random = {}
# for label, count in Counter(y).items():
#     if count == 1:
#         sampling_strategy_random[label] = 50
#     elif count < 50000:
#         sampling_strategy_smote[label] = 50000
#     else:
#         sampling_strategy_smote[label] = count
#
# # Apply RandomOverSampler and SMOTE
# ros = RandomOverSampler(sampling_strategy=sampling_strategy_random, random_state=42)
# X_res, y_res = ros.fit_resample(X, y)
#
# smote = SMOTE(sampling_strategy=sampling_strategy_smote, k_neighbors=1, random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_res, y_res)
# print("Class distribution after resampling:", Counter(y_train_res))
#
# # Encode labels and scale features
# label_encoder = LabelEncoder()
# y_train_res_encoded = label_encoder.fit_transform(y_train_res)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_res)
#
# # Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add channel dimension for CNN
# y_train_tensor = torch.tensor(y_train_res_encoded, dtype=torch.long)
#
# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
#
# # Define the Lightweight CNN-LSTM Model
# class LightweightCNNLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LightweightCNNLSTM, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2)
#
#         # Calculate cnn_output_size based on the input and pooling layers
#         cnn_output_size = input_size // 4
#         self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#
#         # Reshape for LSTM: (batch_size, seq_len, features)
#         x = x.permute(0, 2, 1)
#
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]
#         x = self.fc(x)
#         return x
#
#
# # Instantiate the model, loss function, and optimizer
# input_size = X_train_tensor.shape[2]  # Input size after scaling and adding channel dimension
# hidden_size = 128
# num_classes = len(label_encoder.classes_)
# model = LightweightCNNLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training the model
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
#
# print("Training completed.")


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
