import pandas as pd
import numpy as np
import threading
import time
import pickle
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
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
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

        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def split_data_for_nodes(self, X, y, num_nodes):
        X_train_nodes, y_train_nodes = [], []
        X_split, y_split = np.array_split(X, num_nodes), np.array_split(y, num_nodes)
        for X_part, y_part in zip(X_split, y_split):
            X_train_nodes.append(X_part.reset_index(drop=True))
            y_train_nodes.append(y_part.reset_index(drop=True))
        return X_train_nodes, y_train_nodes

def create_context():
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def encrypt_gradient(gradient, context):
    return ts.ckks_vector(context, gradient.tolist())

def decrypt_gradient(encrypted_gradient):
    return np.array(encrypted_gradient.decrypt())

class FederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, context, learning_rate=0.1, round_num=1):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.context = context
        self.model = None
        self.encrypted_gradient = None
        self.learning_rate = learning_rate * (1 / (1 + 0.01 * round_num))

    def run(self):
        print(f"Node {self.node_id} starting local training with learning rate {self.learning_rate}...")
        if self.X.empty or self.y.empty:
            print(f"Node {self.node_id}: X or y is empty. Skipping training.")
            return

        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=self.learning_rate, random_state=42)
        self.model.fit(self.X, self.y)

        gradients = np.mean([tree.tree_.threshold for tree in self.model.estimators_], axis=0)
        self.encrypted_gradient = encrypt_gradient(gradients, self.context)

        print(f"Node {self.node_id} completed training and encrypted gradients.")

class FederatedLearning:
    def __init__(self, num_rounds=10, learning_rate=0.5):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.global_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.context = create_context()

    def aggregate_gradients(self, encrypted_gradients):
        sum_gradients = sum(encrypted_gradients)
        return sum_gradients / len(encrypted_gradients)

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass=False):
        print("Starting federated learning...")
        start_time = time.time()

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n--- Federated Learning Round {round_num} ---")
            nodes = [FederatedNode(i, X_train_nodes[i], y_train_nodes[i], self.context, self.learning_rate, round_num) for i in range(len(X_train_nodes))]
            for node in nodes:
                node.start()
            for node in nodes:
                node.join()

            encrypted_gradients = [node.encrypted_gradient for node in nodes if node.encrypted_gradient is not None]
            aggregated_encrypted_gradient = self.aggregate_gradients(encrypted_gradients)
            decrypted_gradient = decrypt_gradient(aggregated_encrypted_gradient)
            print(f"Decrypted Aggregated Gradient: {decrypted_gradient}")

            self.global_model.fit(X_train_nodes[0], y_train_nodes[0])
            accuracy = self.global_model.score(X_test, y_test)
            print(f"Federated Model Accuracy After Round {round_num}: {accuracy * 100:.2f}%")
            print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")

            if loop_until_surpass and accuracy > centralized_accuracy:
                print("Federated model accuracy has surpassed centralized accuracy.")
                break

        print(f"Total Federated Learning Training Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    data_path = "smoking.csv"
    target_column = "smoking"
    data_handler = DataHandler(data_path, target_column)
    data = data_handler.load_data()
    X, y = data_handler.preprocess_data(data)
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

    print("Starting centralized training...")
    start_time = time.time()
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    centralized_accuracy = model.score(X_test, y_test)
    print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
    print(f"Centralized Training Time: {time.time() - start_time:.2f} seconds\n")

    X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, 5)
    fl_model = FederatedLearning(num_rounds=5, learning_rate=0.2)
    fl_model.fit(X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass=True)
