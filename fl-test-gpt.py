import numpy as np
import pandas as pd
import pickle
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load or initialize global model
def load_or_initialize_global_model():
    try:
        with open("global_model.pkl", "rb") as f:
            global_model = pickle.load(f)
        print("Loaded existing global model.")
    except FileNotFoundError:
        global_model = LogisticRegression()
        print("No existing global model found, initializing new model.")
    return global_model

# Save global model
def save_global_model(model):
    with open("global_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Global model updated and saved.")

# Preprocess dataset
def preprocess_data(df):
    print("Preprocessing data...")
    target_column = df.columns[-1]  # Assuming last column is the target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

# Encrypt gradients
def encrypt_gradients(gradients, context):
    print("Encrypting gradients...")
    return [ts.ckks_vector(context, grad.tolist()) for grad in gradients]

# Aggregate encrypted gradients using FedAvg
def aggregate_gradients(encrypted_gradients):
    print("Aggregating encrypted gradients...")
    num_nodes = len(encrypted_gradients)
    avg_grad = encrypted_gradients[0]
    for i in range(1, num_nodes):
        avg_grad += encrypted_gradients[i]
    avg_grad /= num_nodes  # Homomorphically averaged
    return avg_grad

# Federated Learning class
class FederatedLearning:
    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        self.global_model = load_or_initialize_global_model()
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
        print("Encryption context created.")

    def fit(self, X_train, y_train, X_test, y_test):
        node_data_size = len(X_train) // self.num_nodes
        X_nodes = [X_train[i * node_data_size: (i + 1) * node_data_size] for i in range(self.num_nodes)]
        y_nodes = [y_train[i * node_data_size: (i + 1) * node_data_size] for i in range(self.num_nodes)]
        
        print("\n--- Federated Learning Round ---")
        gradients_list = []
        
        for i in range(self.num_nodes):
            print(f"Training model at Node {i}...")
            local_model = LogisticRegression()
            local_model.fit(X_nodes[i], y_nodes[i])
            gradients_list.append(local_model.coef_)
            print(f"Node {i} training complete.")
        
        encrypted_gradients = encrypt_gradients(gradients_list, self.context)
        avg_encrypted_grad = aggregate_gradients(encrypted_gradients)
        
        # Decrypting here would be insecure, but needed to update model
        print("Updating global model...")
        decrypted_avg_grad = np.array(avg_encrypted_grad.decrypt())  # In practice, decryption should not be done
        self.global_model.coef_ = decrypted_avg_grad  # Applying gradient update
        save_global_model(self.global_model)
        
        print("Federated round complete.")

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("smoking.csv")  # Provide any dataset with a target variable
    print("Data loaded successfully.")
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    fl_model = FederatedLearning(num_nodes=5)
    fl_model.fit(X_train, y_train, X_test, y_test)
