import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense, Flatten
import numpy as np


# partially understood this part check the super command online
class AttentionLayer(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(1, self.kernel_size, padding='same', activation='sigmoid')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_map = self.conv1(inputs) #creates attention map
        return inputs * attention_map  # Applying the attention map to the input here


# Loading and preprocessing the data (using the edge iiot dataset)
def load_data(filepath):
    print("Loading data...")
    # Load the dataset
    data = pd.read_csv(filepath, encoding='utf-8', delimiter=',')

    # print for column names
    print("Column names in the dataset:", data.columns)

    # uneeded column
    data = data.drop(columns=['frame.time'])

    # NA values = 0
    data = data.fillna(0)

    # uneeded column. dont forget to merge both uneeded lines
    non_numeric_columns = ['ip.src_host', 'ip.dst_host', 'Attack_type']
    data_numeric = data.drop(columns=non_numeric_columns)

    # Converting Attack_label to numerical (attack could be categorical)
    label_encoder = LabelEncoder()
    data['Attack_label'] = label_encoder.fit_transform(data['Attack_label'])

    # Select the features as (X) and target as (y)
    X = data_numeric
    y = data['Attack_label']  # ONLY Assuming 'Attack_label' is the target (check for other labels in the dataset)

    # making sure that all features are numeric (select only numeric columns)
    X = X.select_dtypes(include=['float64', 'int64'])

    # too much difference(normalizing features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Attention-Augmented CNN Model (This is the main algo for local training)
def create_attention_augmented_cnn(input_shape, num_classes):
    # input_shape: the shape of the input data num_classes: the number of output classes
    inputs = layers.Input(shape=input_shape)
    #defines the input layer of the model. It uses Keras' Input layer to specify the shape of the input data

    # Convolutional feature extraction
    x = layers.Reshape((input_shape[0], 1))(inputs)  # Reshape the data for 1D CNN input(some data could be 2D)
    x = layers.Conv1D(32, 3, activation='relu')(x)
    #It applies 32 filters of size 3 to the reshaped input (x).
    # The activation function used is ReLU (Rectified Linear Unit), which helps the model learn non-linear relationships by introducing
    # non-linearity to the output.

    x = layers.MaxPooling1D(2)(x)
    #adds a 1D max-pooling layer with a pool size of 2. Max-pooling helps reduce the dimensionality of the feature maps while retaining
    # important features

    # Attention Layer to enhance features(takes the feature map (x) as input and learns an attention map)
    x = AttentionLayer()(x)

    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    #max poooling reduces the feature map size

    # Apply attention again
    x = AttentionLayer()(x)

    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # flattens the 3D feature map into a 1D vector
    x = Flatten()(x)

    #  fully connected layer is added with 256 neurons
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    #num_classes neurons (one for each possible class in classification)
    #The softmax activation function is used here, which converts the raw
    # outputs into probabilities, with each value representing the likelihood of the sample belonging to a particular class.

    model = models.Model(inputs, x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Train and evaluate phase (/////training is done here/////)
def train_and_evaluate(X, y, num_clients, global_model, epochs=5):
    print("Splitting data into training and testing sets...")
    # Split the data into training and testing sets (dataset was too big)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Reshaping data for CNN input (adding a single channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Simulation for federated learning ( weights are aggregated after each training)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # array to store client models after local training
        local_models = []

        # Simulate local training on each client(////smaller client training////)
        for client_id in range(num_clients):
            print(f"Training on Client {client_id + 1}")

            # Copying the global model to ensure smaller client models have same architecture and weights
            local_model = create_attention_augmented_cnn(X_train.shape[1:], len(set(y)))
            local_model.set_weights(global_model.get_weights())  # Ensure same weights

            # local training
            local_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

            # Append the local model to the local_model array
            local_models.append(local_model)

        # Average the weights from all local models(used fedavg for this)
        # must ensure all weights have the same shape and are compatible
        local_weights = [model.get_weights() for model in local_models]

        # Used np.mean here to average the weights(gpt advice , search later)
        global_weights = [np.mean(np.array([w[i] for w in local_weights]), axis=0) for i in
                          range(len(local_weights[0]))]

        # changing the global models weights to the newly averaged weights
        global_model.set_weights(global_weights)

        # Evaluate the global model using some data from dataset(test data)
        print("Evaluating the global model...")
        global_model.evaluate(X_test, y_test, verbose=1)

        print(f"Epoch {epoch + 1} completed.")

    # Final evaluation
    print("\nFinal model evaluation:")
    y_pred = global_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
    print(classification_report(y_test, y_pred))


# Main function to load data and train the model
def main(filepath, num_clients=5, epochs=5):
    print("Starting data processing and model training...")
    X, y = load_data(filepath)

    # Creating the global model
    global_model = create_attention_augmented_cnn(X.shape[1:], len(set(y)))

    # Train the model with federated learning
    train_and_evaluate(X, y, num_clients, global_model, epochs)


# path to dataset
filepath = "C:/Users/sives/OneDrive/Documents/proj papers/phase 2/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

if __name__ == "__main__":
    main(filepath)