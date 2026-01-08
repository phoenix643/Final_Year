import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Load the dataset
    data = pd.read_csv(filepath, encoding='utf-8', delimiter=',')

    # Debugging print to check the column names
    print("Column names in the dataset:", data.columns)

    # Drop 'frame.time' as it's not used for model training
    data = data.drop(columns=['frame.time'])

    # Handle missing values (fill with 0)
    data = data.fillna(0)

    # Remove non-numeric columns (IP addresses and Attack Type columns)
    non_numeric_columns = ['ip.src_host', 'ip.dst_host', 'Attack_type']
    data_numeric = data.drop(columns=non_numeric_columns)

    # Convert Attack_label to numeric (if it's categorical)
    label_encoder = LabelEncoder()
    data['Attack_label'] = label_encoder.fit_transform(data['Attack_label'])

    # Select features (X) and target (y)
    X = data_numeric
    y = data['Attack_label']  # Assuming 'Attack_label' is the target

    # Ensure all features are numeric by selecting only numeric columns
    X = X.select_dtypes(include=['float64', 'int64'])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Attention-Augmented CNN Model
def create_attention_augmented_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Convolutional feature extraction
    x = layers.Reshape((input_shape[0], 1))(inputs)  # Reshape for 1D CNN input
    x = layers.Conv1D(32, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # Attention Layer to enhance features
    x = AttentionLayer()(x)

    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # Apply attention again
    x = AttentionLayer()(x)

    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # Flatten for fully connected layers
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)  # Softmax for classification

    model = models.Model(inputs, x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Train and evaluate model
def train_and_evaluate(X, y, num_clients, global_model, epochs=5):
    print("Splitting data into training and testing sets...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Reshape data for CNN input (adding a single channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Simulate federated learning: train on multiple clients and aggregate the model weights
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # List to store client models after local training
        local_models = []

        # Simulate local training on each client
        for client_id in range(num_clients):
            print(f"Training on Client {client_id + 1}")

            # Clone the global model to ensure all models have the same architecture and weights
            local_model = create_attention_augmented_cnn(X_train.shape[1:], len(set(y)))
            local_model.set_weights(global_model.get_weights())  # Ensure same weights

            # Train the model locally
            local_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

            # Append the local model to the list
            local_models.append(local_model)

        # Federated Averaging: Average the weights from all local models
        # Ensure that all weights have the same shape and are compatible
        local_weights = [model.get_weights() for model in local_models]

        # Use np.mean to average the weights
        global_weights = [np.mean(np.array([w[i] for w in local_weights]), axis=0) for i in
                          range(len(local_weights[0]))]

        # Set the global model's weights to the averaged weights
        global_model.set_weights(global_weights)

        # Evaluate the global model on the test data
        print("Evaluating the global model...")
        global_model.evaluate(X_test, y_test, verbose=1)

        print(f"Epoch {epoch + 1} completed.")

    # Final evaluation
    print("\nFinal model evaluation:")
    y_pred = global_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Main function to load data and train the model
def main(filepath, num_clients=5, epochs=5):
    print("Starting data processing and model training...")
    X, y = load_data(filepath)

    # Create the global model
    global_model = create_attention_augmented_cnn(X.shape[1:], len(set(y)))

    # Train the model with federated learning
    train_and_evaluate(X, y, num_clients, global_model, epochs)


# Provide the file path to your dataset
filepath = "C:/Users/sives/OneDrive/Documents/proj papers/phase 2/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

if __name__ == "__main__":
    main(filepath)





