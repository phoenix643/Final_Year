import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense, Flatten
import numpy as np


# Attention Mechanism: Spatial Attention
class AttentionLayer(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(1, self.kernel_size, padding='same', activation='sigmoid')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_map = self.conv1(inputs)
        return inputs * attention_map  # Apply the attention map to the input


# Load and preprocess data
def load_data(filepath):
    print("Loading data...")
    data = pd.read_csv(filepath, encoding='utf-8', delimiter=',')
    print("Column names in the dataset:", data.columns)
    data = data.drop(columns=['frame.time'])
    data = data.fillna(0)
    non_numeric_columns = ['ip.src_host', 'ip.dst_host', 'Attack_type']
    data_numeric = data.drop(columns=non_numeric_columns)
    label_encoder = LabelEncoder()
    data['Attack_label'] = label_encoder.fit_transform(data['Attack_label'])
    X = data_numeric.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = data['Attack_label']
    return X_scaled, y


# Attention-Augmented CNN Model
def create_attention_augmented_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.Conv1D(32, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = AttentionLayer()(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = AttentionLayer()(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Federated Averaging (FedAvg) Algorithm
def federated_averaging(global_model, local_models):
    local_weights = [model.get_weights() for model in local_models]
    global_weights = [np.mean(np.array([w[i] for w in local_weights]), axis=0) for i in range(len(local_weights[0]))]
    global_model.set_weights(global_weights)


# Train and evaluate model with Federated Learning
def train_and_evaluate(X, y, num_clients, global_model, epochs=5):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        local_models = []
        for client_id in range(num_clients):
            print(f"Training on Client {client_id + 1}")
            local_model = create_attention_augmented_cnn(X_train.shape[1:], len(set(y)))
            local_model.set_weights(global_model.get_weights())
            local_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
            local_models.append(local_model)

        federated_averaging(global_model, local_models)
        print("Evaluating the global model...")
        global_model.evaluate(X_test, y_test, verbose=1)
        print(f"Epoch {epoch + 1} completed.")

    print("\nFinal model evaluation:")
    y_pred_prob = global_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print(classification_report(y_test, y_pred))

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)  # Adjust pos_label if needed
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Main function to load data and train the model
def main(filepath, num_clients=5, epochs=5):
    print("Starting data processing and model training...")
    X, y = load_data(filepath)
    global_model = create_attention_augmented_cnn(X.shape[1:], len(set(y)))
    train_and_evaluate(X, y, num_clients, global_model, epochs)


# Provide the file path to your dataset
filepath = "C:/Users/sives/OneDrive/Documents/proj papers/phase 2/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

if __name__ == "__main__":
    main(filepath)
