import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Custom AGC Optimizer class
class AGCOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, clip_factor=0.01, **kwargs):
        super(AGCOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.clip_factor = clip_factor

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'grad')

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        for grad, var in grads_and_vars:
            grad_norm = tf.norm(grad)
            var_norm = tf.norm(var)
            clip_value = self.clip_factor * var_norm
            grad = tf.clip_by_value(grad, -clip_value, clip_value)

            self.optimizer.apply_gradients([(grad, var)])

        return self.optimizer.apply_gradients(grads_and_vars)


# Load and preprocess data
def load_data(filepath):
    data = pd.read_csv(filepath, encoding='utf-8', delimiter=',')
    print("Dataset loaded with columns:", data.columns)

    # Reduce dataset to 30% for faster runtime
    data = data.sample(frac=0.3, random_state=42)

    # Handle categorical features and labels
    label_encoder = LabelEncoder()
    data['Attack_label'] = label_encoder.fit_transform(data['Attack_label'])
    X = data.drop(columns=['Attack_label', 'Attack_type'])
    y = data['Attack_label']

    # Normalize numerical columns if necessary
    X = (X - X.mean()) / X.std()

    return X, y


# Model definition (simple neural network example)
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# Federated learning training and evaluation
def train_and_evaluate(X, y, num_clients, global_model, epochs):
    # Split data into "clients" (subsets of data for federated learning)
    client_data = np.array_split(X, num_clients)
    client_labels = np.array_split(y, num_clients)

    print("Starting federated learning...")

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        for client_idx in range(num_clients):
            print(f'Training on Client {client_idx + 1}')

            client_X = client_data[client_idx]
            client_y = client_labels[client_idx]

            # Create a fresh copy of the global model for each client
            client_model = tf.keras.models.clone_model(global_model)
            client_model.set_weights(global_model.get_weights())

            # Compile the model with AGC optimizer
            agc_optimizer = AGCOptimizer(tf.keras.optimizers.Adam(learning_rate=0.001), clip_factor=0.01)
            client_model.compile(optimizer=agc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model on the client's data
            print(f"Training client {client_idx + 1} for 1 epoch")
            client_model.fit(client_X, client_y, epochs=1, batch_size=32, verbose=1)

            # Aggregate weights back to the global model
            global_weights = global_model.get_weights()
            client_weights = client_model.get_weights()
            updated_weights = [global_w * 0.9 + client_w * 0.1 for global_w, client_w in
                               zip(global_weights, client_weights)]
            global_model.set_weights(updated_weights)

        # Evaluate global model after each epoch
        print(f"Evaluating global model after epoch {epoch + 1}...")
        evaluate_global_model(global_model, X, y)


# Evaluation function
def evaluate_global_model(global_model, X, y):
    y_pred = global_model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels
    accuracy = accuracy_score(y, y_pred)
    print(f'Global model accuracy: {accuracy:.4f}')


# Main function
def main(filepath, epochs=3, num_clients=3):
    print("Starting data processing and model training...")

    # Load data
    X, y = load_data(filepath)

    # Split data into training and testing sets (50% training, 50% testing of the 30% subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    print("Building the global model...")
    # Build global model
    global_model = build_model((X_train.shape[1],))
    global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                         metrics=['accuracy'])

    print("Starting federated learning process...")
    # Start federated learning
    train_and_evaluate(X_train, y_train, num_clients, global_model, epochs)

    # Evaluate the global model on test data
    print("\nEvaluating the global model on test data...")
    evaluate_global_model(global_model, X_test, y_test)


if __name__ == "__main__":
    filepath = "C:\\Users\\sives\\OneDrive\\Documents\\proj papers\\phase 2\\Edge-IIoTset dataset\\Selected dataset for ML and DL\\DNN-EdgeIIoT-dataset.csv"  # Change to your dataset path
    main(filepath)
