"""
this programm is beta that is generalized to take any dataset 

"""



import pandas as pd
import numpy as np
import threading
import time
import pickle
import tenseal as ts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
    def __init__(self, node_id, X, y, global_weights, context, learning_rate=0.1):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.global_weights = global_weights
        self.context = context
        self.local_model_weights = None
        self.learning_rate = learning_rate

    def run(self):
        print(f"Node {self.node_id} starting local training...")
        if self.X.empty or self.y.empty:
            print(f"Node {self.node_id}: X or y is empty. Skipping training.")
            return

        model = LogisticRegression(fit_intercept=True, max_iter=300, solver='saga', penalty='l2', C=1.0, warm_start=True)
        coef = self.global_weights[:-1].reshape(1, -1)
        intercept = self.global_weights[-1:]

        model.coef_ = coef
        model.intercept_ = intercept

        try:
            model.fit(self.X, self.y)
        except ValueError as e:
            print(f"Node {self.node_id}: Error during model.fit: {e}")
            return

        local_weights = np.concatenate([model.coef_.flatten(), model.intercept_])
        self.local_model_weights = encrypt_weights(local_weights, self.context)
        print(f"Node {self.node_id} completed local training and stored encrypted model weights.")

class FederatedLearning:
    def __init__(self, num_rounds=10, learning_rate=0.5, model_path="global_model.pkl"):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.model_path = model_path

    def initialize_global_model(self, n_features):
        return np.random.randn(n_features + 1) * 0.01  # Use random initialization for stability

    def load_global_model(self, n_features):
        try:
            with open(self.model_path, 'rb') as f:
                global_weights = pickle.load(f)
            print("Loaded global model weights from file.")
        except FileNotFoundError:
            global_weights = self.initialize_global_model(n_features)
            print("No saved model weights found. Initialized new global model.")
        return global_weights

    def save_global_model(self, global_weights):
        with open(self.model_path, 'wb') as f:
            pickle.dump(global_weights, f)
        print("Saved global model weights to file.")

    def aggregate_model_weights(self, model_weights_list, X_train_nodes, context):
        total_samples = sum(len(X) for X in X_train_nodes)
        aggregated_weights = ts.ckks_vector(context, np.zeros_like(decrypt_weights(model_weights_list[0])))

        for i, weights in enumerate(model_weights_list):
            if weights is not None:
                num_samples = len(X_train_nodes[i])
                weight_fraction = num_samples / total_samples
                weighted_encrypted = weights * weight_fraction
                aggregated_weights += weighted_encrypted

        return aggregated_weights

    def update_global_model(self, global_weights, aggregated_weights):
        decrypted_weights = decrypt_weights(aggregated_weights)
        print("Decrypted Aggregated Weights:", decrypted_weights)  # Print decrypted weights
        global_weights[:-1] = decrypted_weights[:-1]
        global_weights[-1] = decrypted_weights[-1]
        return global_weights

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass=False):
        print("Starting federated learning...")
        start_time = time.time()

        n_features = X_train_nodes[0].shape[1]
        global_weights = self.load_global_model(n_features)
        context = create_context()

        round_num = 0
        while True:
            round_num += 1
            print(f"\n--- Federated Learning Round {round_num} ---")
            nodes = []
            model_weights_list = []

            for i in range(len(X_train_nodes)):
                node = FederatedNode(i, X_train_nodes[i], y_train_nodes[i], global_weights, context, learning_rate=self.learning_rate * (1 / (1 + 0.01 * round_num)))
                nodes.append(node)
                node.start()

            for node in nodes:
                node.join()
                model_weights_list.append(node.local_model_weights)

            aggregated_weights = self.aggregate_model_weights(model_weights_list, X_train_nodes, context)
            global_weights = self.update_global_model(global_weights, aggregated_weights)

            # Check the model accuracy
            model = LogisticRegression(max_iter=2000, solver='saga')
            model.coef_ = global_weights[:-1].reshape(1, -1)
            model.intercept_ = global_weights[-1:]
            model.fit(X_train_nodes[0], y_train_nodes[0])  # Fit with a small sample to set classes_
            accuracy = model.score(X_test, y_test)
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

        self.save_global_model(global_weights)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Federated Learning Training Time: {training_time:.2f} seconds\n")
        return model



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
        model = LogisticRegression(max_iter=1000, penalty='l2', solver='saga')
        
        print("Fitting centralized model...")
        model.fit(X_train, y_train)
        
        centralized_accuracy = model.score(X_test, y_test)
        print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        print(f"Centralized Training Time: {time.time() - start_time:.2f} seconds\n")

        # Federated Learning
        print("Preparing federated learning...")
        num_nodes = 1
        X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, num_nodes)

        fl_model = FederatedLearning(num_rounds=10, learning_rate=0.2, model_path="global_model.pkl")
        
        # Set the flag to loop until surpassing centralized accuracy
        loop_until_surpass = False
        
        print("Starting federated learning...")
        fl_model.fit(X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass)
        
    except Exception as e:
        print(f"An error occurred: {e}")

