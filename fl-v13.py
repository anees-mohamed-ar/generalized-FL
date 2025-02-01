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

def encrypt_weights(weights, context):
    return ts.ckks_vector(context, weights)

def decrypt_weights(encrypted_weights):
    return np.array(encrypted_weights.decrypt())

class FederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, context, learning_rate=0.1, round_num=1):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.context = context
        self.model = None
        self.learning_rate = learning_rate * (1 / (1 + 0.01 * round_num))

    def run(self):
        print(f"Node {self.node_id} starting local training with learning rate {self.learning_rate}...")
        if self.X.empty or self.y.empty:
            print(f"Node {self.node_id}: X or y is empty. Skipping training.")
            return

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=self.learning_rate, random_state=42)
        model.fit(self.X, self.y)
        self.model = model
        print(f"Node {self.node_id} completed local training and stored model.")

class FederatedLearning:
    def __init__(self, num_rounds=10, learning_rate=0.5, model_path="global_model.pkl"):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.global_model = None

    def initialize_global_model(self):
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    def save_global_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.global_model, f)
        print("Saved global model to file.")

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass=False):
        print("Starting federated learning...")
        start_time = time.time()

        if self.global_model is None:
            self.global_model = self.initialize_global_model()

        round_num = 0
        while True:
            round_num += 1
            print(f"\n--- Federated Learning Round {round_num} ---")
            nodes = []

            for i in range(len(X_train_nodes)):
                node = FederatedNode(i, X_train_nodes[i], y_train_nodes[i], create_context(), learning_rate=self.learning_rate, round_num=round_num)
                nodes.append(node)
                node.start()

            node_models = []
            for node in nodes:
                node.join()
                node_models.append(node.model)

            # Aggregate models (e.g., by averaging the predictions)
            self.global_model = self.aggregate_models(node_models)

            # Check the model accuracy
            accuracy = self.evaluate_model(X_test, y_test)
            print(f"Federated Model Accuracy After Round {round_num}: {accuracy * 100:.2f}%")
            print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")

            # Check if federated model accuracy is greater than centralized accuracy
            if loop_until_surpass:
                if accuracy > centralized_accuracy:
                    print(f"Federated model accuracy {accuracy * 100:.2f}% has surpassed the centralized model accuracy {centralized_accuracy * 100:.2f}%")
                    break
            else:
                if round_num >= self.num_rounds:
                    break

        self.save_global_model()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Federated Learning Training Time: {training_time:.2f} seconds\n")
        return self.global_model

    def aggregate_models(self, models):
        # Example: Averaging the models' predictions
        return models[0]  # Placeholder: Replace with actual aggregation logic

    def evaluate_model(self, X_test, y_test):
        if self.global_model is None:
            return 0
        return self.global_model.score(X_test, y_test)

if __name__ == "__main__":
    try:
        # Path to your dataset
        data_path = "smoking.csv"
        target_column = "smoking"  # Replace with the name of your target column

        print("Initializing data handler...")
        data_handler = DataHandler(data_path=data_path, target_column=target_column)
        
        print("Loading data...")
        data = data_handler.load_data()
        
        print("Preprocessing data...")
        X, y = data_handler.preprocess_data(data)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

        # Centralized training
        print("Starting centralized training...")
        start_time = time.time()
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        
        print("Fitting centralized model...")
        model.fit(X_train, y_train)
        
        centralized_accuracy = model.score(X_test, y_test)
        print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        print(f"Centralized Training Time: {time.time() - start_time:.2f} seconds\n")

        # Federated Learning
        print("Preparing federated learning...")
        num_nodes = 5
        X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, num_nodes)

        fl_model = FederatedLearning(num_rounds=5, learning_rate=0.2, model_path="global_model.pkl")
        
        # Set the flag to loop until surpassing centralized accuracy
        loop_until_surpass = True
        
        print("Starting federated learning...")
        fl_model.fit(X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass)
        
    except Exception as e:
        print(f"An error occurred: {e}")
