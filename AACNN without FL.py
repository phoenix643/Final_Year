import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
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


# Train and evaluate the model for a single client
def train_and_evaluate_single_client(X, y, epochs=5):
    print("Splitting data into training and testing sets...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Reshape data for CNN input (adding a single channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create the model
    model = create_attention_augmented_cnn(X_train.shape[1:], len(set(y)))

    # Train the model on the single client's data
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    # Evaluate the model on the test data
    print("\nEvaluating the model...")
    model.evaluate(X_test, y_test, verbose=1)

    # Final evaluation with classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
    print(classification_report(y_test, y_pred))


# Main function to load data and train the model
def main(filepath, epochs=5):
    print("Starting data processing and model training...")
    X, y = load_data(filepath)

    # Train the model on a single client
    train_and_evaluate_single_client(X, y, epochs)


# Provide the file path to your dataset
filepath = "C:/Users/sives/OneDrive/Documents/proj papers/phase 2/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

if __name__ == "__main__":
    main(filepath)
