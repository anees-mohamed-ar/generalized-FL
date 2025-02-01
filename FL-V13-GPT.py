import pandas as pd
import numpy as np
import threading
import time
import pickle
import os
import tenseal as ts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DataHandler:
    def __init__(self, data_path, target_column, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        print("Loading data...")
        data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")
        return data

    def preprocess_data(self, data):
        print("Preprocessing data...")
        data = data.replace('?', np.nan)
        data = shuffle(data, random_state=self.random_state)

        imputer = SimpleImputer(strategy='most_frequent')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                label_encoders[col] = encoder

        target_encoder = LabelEncoder()
        data[self.target_column] = target_encoder.fit_transform(data[self.target_column])

        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        print("Data preprocessing completed.")
        return X, y

    def split_data(self, X, y):
        print("Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def split_data_for_nodes(self, X, y, num_nodes):
        print(f"Splitting data into {num_nodes} nodes...")
        X_split, y_split = np.array_split(X, num_nodes), np.array_split(y, num_nodes)
        return [X_part.reset_index(drop=True) for X_part in X_split], [y_part.reset_index(drop=True) for y_part in y_split]

def create_context():
    print("Creating CKKS encryption context...")
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    context.generate_relin_keys()
    print("Encryption context created.")
    return context

def encrypt_gradients(gradients, context):
    print("Encrypting gradients...")
    encrypted = [ts.ckks_vector(context, grad) for grad in gradients]
    print("Gradients encrypted successfully.")
    return encrypted

def decrypt_gradients(encrypted_gradients):
    print("Decrypting gradients...")
    decrypted = [np.array(grad.decrypt()) for grad in encrypted_gradients]
    print("Gradients decrypted successfully.")
    return decrypted

def average_gradients(gradients):
    print("Averaging gradients...")
    return np.mean(gradients, axis=0)

class FederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, context, learning_rate=0.1):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.context = context
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, random_state=42)
        self.gradients = None

    def run(self):
        if self.X.empty or self.y.empty:
            print(f"Node {self.node_id} has no data, skipping...")
            return

        print(f"Training model at Node {self.node_id}...")
        self.model.fit(self.X, self.y)
        print(f"Node {self.node_id} training complete.")

        params = self.model.get_params()
        numeric_params = [value for key, value in params.items() if isinstance(value, (int, float, np.number))]
        self.gradients = np.array(numeric_params)

class FederatedLearning:
    def __init__(self, num_rounds=10, learning_rate=0.1, model_path="global_model.pkl"):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.global_model = self.load_global_model()
        self.context = create_context()

    def initialize_global_model(self):
        print("Initializing global model...")
        return GradientBoostingClassifier(n_estimators=100, learning_rate=self.learning_rate, random_state=42)

    def save_global_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.global_model, f)
        print("Saved global model.")

    def load_global_model(self):
        if os.path.exists(self.model_path):
            print("Loading existing global model...")
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        print("No existing global model found, initializing new model.")
        return self.initialize_global_model()

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy):
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n--- Federated Learning Round {round_num} ---")
            nodes = [FederatedNode(i, X_train_nodes[i], y_train_nodes[i], self.context, self.learning_rate) for i in range(len(X_train_nodes))]

            for node in nodes:
                node.start()
            for node in nodes:
                node.join()

            node_gradients = [node.gradients for node in nodes if node.gradients is not None]

            if not node_gradients:
                print("No valid gradients received, skipping update.")
                continue

            encrypted_gradients = encrypt_gradients(node_gradients, self.context)
            avg_encrypted_gradients = average_gradients(encrypted_gradients)
            avg_gradients = decrypt_gradients([avg_encrypted_gradients])[0]

            self.global_model.set_params(**dict(zip(self.global_model.get_params().keys(), avg_gradients)))

            self.save_global_model()

            accuracy = self.evaluate_model(X_test, y_test)
            print(f"Federated Model Accuracy After Round {round_num}: {accuracy * 100:.2f}%")

            if accuracy > centralized_accuracy:
                print("Federated model surpassed centralized model accuracy.")
                break

    def evaluate_model(self, X_test, y_test):
        print("Evaluating global model...")
        return self.global_model.score(X_test, y_test)

if __name__ == "__main__":
    data_path = "smoking.csv"
    target_column = "smoking"

    data_handler = DataHandler(data_path=data_path, target_column=target_column)
    data = data_handler.load_data()
    X, y = data_handler.preprocess_data(data)
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    centralized_accuracy = model.score(X_test, y_test)

    num_nodes = 5
    X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, num_nodes)

    fl_model = FederatedLearning(num_rounds=10, learning_rate=0.2, model_path="global_model.pkl")
    fl_model.fit(X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy)
