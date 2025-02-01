import pandas as pd
import numpy as np
import threading
import time
import json
import tenseal as ts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit  # For sigmoid function

class DataHandler:
    def __init__(self, data_path, target_column, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None

    def load_data(self):
        return pd.read_csv(self.data_path)

    def preprocess_data(self, data):
        data = data.replace('?', np.nan)
        data = shuffle(data, random_state=self.random_state)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Encode categorical features
        for col in data.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self.label_encoders[col] = encoder

        # Encode target variable
        self.target_encoder = LabelEncoder()
        data[self.target_column] = self.target_encoder.fit_transform(data[self.target_column])

        # Split features and target
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        # Scale features
        X = pd.DataFrame(self.feature_scaler.fit_transform(X), columns=X.columns)

        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def split_data_for_nodes(self, X, y, num_nodes):
        """Split data into portions for each node, returning numpy arrays."""
        X_split = np.array_split(X, num_nodes)
        y_split = np.array_split(y, num_nodes)
        return [x.reset_index(drop=True) for x in X_split], [y.reset_index(drop=True) for y in y_split]

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_dim, 64) * 0.01
        self.b1 = np.zeros((64,))
        self.W2 = np.random.randn(64, 32) * 0.01
        self.b2 = np.zeros((32,))
        self.W3 = np.random.randn(32, 1) * 0.01
        self.b3 = np.zeros((1,))

    def forward_propagation(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)  # ReLU activation
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = expit(self.Z3)  # Sigmoid activation
        return self.A3

    def backward_propagation(self, X, y, output):
        m = X.shape[0]
        
        # Compute gradients
        dZ3 = output - y.reshape(-1, 1)
        dW3 = (1/m) * np.dot(self.A2.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0)
        
        dZ2 = np.dot(dZ3, self.W3.T) * (self.Z2 > 0)  # ReLU derivative
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.Z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def get_gradients(self, X, y):
        # Forward pass
        output = self.forward_propagation(X)
        # Backward pass
        return self.backward_propagation(X, y, output)

    def apply_gradients(self, gradients):
        # Apply gradient updates with learning rate
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
        self.W3 -= self.learning_rate * gradients['dW3']
        self.b3 -= self.learning_rate * gradients['db3']

    def predict(self, X):
        probabilities = self.forward_propagation(X)
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions.flatten() == y)

# Also update the FederatedNode class to handle numpy arrays
class FederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, input_dim, context, learning_rate=0.01):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X  # Now expecting numpy array
        self.y = y  # Now expecting numpy array
        self.context = context
        self.model = NeuralNetwork(input_dim=input_dim, learning_rate=learning_rate)
        self.gradients = None

    def run(self):
        print(f"Node {self.node_id} starting local training...")
        if len(self.X) == 0 or len(self.y) == 0:
            print(f"Node {self.node_id}: Empty dataset. Skipping training.")
            return

        # Compute gradients (X and y are already numpy arrays)
        self.gradients = self.model.get_gradients(self.X, self.y)
        print(f"Node {self.node_id} completed local gradient computation.")

def create_context():
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def encrypt_gradients(gradients, context):
    encrypted_grads = {}
    for key, grad in gradients.items():
        encrypted_grads[key] = ts.ckks_vector(context, grad.flatten())
    return encrypted_grads

def decrypt_gradients(encrypted_gradients):
    decrypted_grads = {}
    for key, grad in encrypted_gradients.items():
        decrypted_grads[key] = np.array(grad.decrypt())
    return decrypted_grads

def average_gradients(encrypted_gradients_list, num_nodes):
    # Initialize with the first node's gradients
    avg_gradients = encrypted_gradients_list[0]
    
    # Add gradients from other nodes
    for grads in encrypted_gradients_list[1:]:
        for key in avg_gradients:
            avg_gradients[key] += grads[key]
    
    # Average the gradients
    for key in avg_gradients:
        avg_gradients[key] = avg_gradients[key] * (1.0 / num_nodes)
    
    return avg_gradients

def clip_gradients(gradients, max_norm):
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(np.square(grad))
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scaling_factor = max_norm / total_norm
        for key in gradients:
            gradients[key] *= scaling_factor
    
    return gradients

class FederatedLearning:
    def __init__(self, input_dim, num_rounds=10, learning_rate=0.01, model_path="global_model.json"):
        self.input_dim = input_dim
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.global_model = NeuralNetwork(input_dim=input_dim, learning_rate=learning_rate)
        self.context = create_context()

    def save_global_model(self):
        """Save the global model parameters to a JSON file."""
        try:
            model_data = {
                'W1': self.global_model.W1.tolist(),
                'b1': self.global_model.b1.tolist(),
                'W2': self.global_model.W2.tolist(),
                'b2': self.global_model.b2.tolist(),
                'W3': self.global_model.W3.tolist(),
                'b3': self.global_model.b3.tolist()
            }
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f)
            print("Saved global model to file.")
        except Exception as e:
            print(f"Error saving global model: {e}")

    def load_global_model(self):
        """Load the global model parameters from a JSON file."""
        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            self.global_model.W1 = np.array(model_data['W1'])
            self.global_model.b1 = np.array(model_data['b1'])
            self.global_model.W2 = np.array(model_data['W2'])
            self.global_model.b2 = np.array(model_data['b2'])
            self.global_model.W3 = np.array(model_data['W3'])
            self.global_model.b3 = np.array(model_data['b3'])
            print("Loaded global model from file.")
        except FileNotFoundError:
            print("No existing model found. Starting with a new model.")
        except Exception as e:
            print(f"Error loading global model: {e}")
            print("Starting with a new model.")

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass=False):
        print("Starting federated learning...")
        start_time = time.time()
        self.load_global_model()

        best_accuracy = 0
        round_num = 0
        max_rounds = 1000  # Safety limit to prevent infinite loops
        
        while True:
            round_num += 1
            if round_num > max_rounds:
                print(f"Reached maximum rounds ({max_rounds}) without surpassing centralized accuracy.")
                break
                
            print(f"\n--- Federated Learning Round {round_num} ---")
            
            # Create and start nodes
            nodes = []
            for i in range(len(X_train_nodes)):
                node = FederatedNode(i, X_train_nodes[i], y_train_nodes[i], 
                                   self.input_dim, self.context, self.learning_rate)
                nodes.append(node)
                node.start()

            # Wait for all nodes to complete and collect gradients
            node_gradients = []
            for node in nodes:
                node.join()
                node_gradients.append(encrypt_gradients(node.gradients, self.context))

            # Average encrypted gradients
            avg_encrypted_gradients = average_gradients(node_gradients, len(nodes))

            # Decrypt averaged gradients
            avg_gradients = decrypt_gradients(avg_encrypted_gradients)

            # Clip gradients
            avg_gradients = clip_gradients(avg_gradients, max_norm=1.0)

            # Reshape gradients back to original shapes
            reshaped_gradients = {
                'dW1': avg_gradients['dW1'].reshape(self.global_model.W1.shape),
                'db1': avg_gradients['db1'].reshape(self.global_model.b1.shape),
                'dW2': avg_gradients['dW2'].reshape(self.global_model.W2.shape),
                'db2': avg_gradients['db2'].reshape(self.global_model.b2.shape),
                'dW3': avg_gradients['dW3'].reshape(self.global_model.W3.shape),
                'db3': avg_gradients['db3'].reshape(self.global_model.b3.shape)
            }

            # Update global model
            self.global_model.apply_gradients(reshaped_gradients)

            # Evaluate model
            accuracy = self.global_model.score(X_test, y_test)
            print(f"Round {round_num} - Federated Model Accuracy: {accuracy * 100:.2f}%")
            print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")

            # Save if it's the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_global_model()

            # Check if we should continue training
            if loop_until_surpass:
                if accuracy > centralized_accuracy:
                    print(f"\nFederated model has surpassed centralized model after {round_num} rounds!")
                    break
            else:
                if round_num >= self.num_rounds:
                    break

            # Adaptive learning rate
            self.learning_rate *= 0.95  # Reduce learning rate over time

        training_time = time.time() - start_time
        print(f"\nFederated Learning completed in {training_time:.2f} seconds")
        print(f"Best accuracy achieved: {best_accuracy * 100:.2f}%")
        return self.global_model

def main():
    try:
        # Set the flag for training until surpassing centralized model
        loop_until_surpass = True
        
        # Initialize data handler
        data_path = "diabetesData.csv"
        target_column = "target"
        data_handler = DataHandler(data_path=data_path, target_column=target_column)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = data_handler.load_data()
        X, y = data_handler.preprocess_data(data)
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

        # Convert pandas objects to numpy arrays
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values

        # Train centralized model for comparison
        print("Training centralized model...")
        start_time = time.time()
        centralized_model = NeuralNetwork(input_dim=X_train_np.shape[1], learning_rate=0.01)
        
        # Train for a few epochs
        num_epochs = 10
        batch_size = 32
        n_samples = X_train_np.shape[0]
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_np[indices]
            y_shuffled = y_train_np[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                gradients = centralized_model.get_gradients(batch_X, batch_y)
                centralized_model.apply_gradients(gradients)
        
        centralized_time = time.time() - start_time
        centralized_accuracy = centralized_model.score(X_test_np, y_test_np)
        print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        print(f"Centralized Training Time: {centralized_time:.2f} seconds\n")

        # Federated Learning Setup
        print("Setting up federated learning...")
        num_nodes = 5
        X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, num_nodes)
        
        # Convert to numpy arrays
        X_train_nodes = [X.values for X in X_train_nodes]
        y_train_nodes = [y.values for y in y_train_nodes]

        # Initialize federated learning
        fl_model = FederatedLearning(
            input_dim=X_train_np.shape[1],
            num_rounds=50,
            learning_rate=0.01,
            model_path="federated_model.json"
        )
        
        # Train federated model with surpass flag
        print("Starting federated learning training...")
        federated_model = fl_model.fit(
            X_train_nodes,
            y_train_nodes,
            X_test_np,
            y_test_np,
            centralized_accuracy,
            loop_until_surpass=loop_until_surpass
        )

        # Final evaluation
        federated_accuracy = federated_model.score(X_test_np, y_test_np)
        print("\nFinal Results:")
        print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        print(f"Federated Model Accuracy: {federated_accuracy * 100:.2f}%")

        # Compare models
        if federated_accuracy > centralized_accuracy:
            improvement = ((federated_accuracy - centralized_accuracy) / centralized_accuracy) * 100
            print(f"\nFederated learning achieved {improvement:.2f}% improvement over centralized learning")
        else:
            difference = ((centralized_accuracy - federated_accuracy) / centralized_accuracy) * 100
            print(f"\nFederated learning performed {difference:.2f}% worse than centralized learning")

        # Save final model
        fl_model.save_global_model()
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()