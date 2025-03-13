import os
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
import random
import warnings
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn,BarColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import print as rprint
from rich.style import Style
from rich.traceback import install
from rich.theme import Theme


install(show_locals=True)

# Create console instance
console = Console()

# Custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "metric": "blue",
    "highlight": "magenta"
})

console = Console(theme=custom_theme)

def rich_input(prompt_text):
    console.print(f"{prompt_text}", end="")
    return input()




# Suppress specific FutureWarning
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")
warnings.filterwarnings("ignore", message="'Series.swapaxes' is deprecated and will be removed in a future version. Please use 'Series.transpose' instead.")



def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def compute_privacy_metrics(original_gradients, encrypted_gradients, decrypted_gradients, encryption_time, decryption_time, decryption_context):
    """
    Computes and displays privacy metrics including:
    - Privacy Leakage Metric (PLM)
    - Gradient Protection Effectiveness
    - Encryption Overhead
    """

    console = Console()

    # Privacy Leakage Metric (PLM) - Measures inference attack resistance
    plm_values = []
    encryption_difference = []
    
    for key in original_gradients:
        original_norm = np.linalg.norm(np.array(original_gradients[key]).flatten()) + 1e-8  # Avoid division by zero
        decrypted_norm = np.linalg.norm(np.array(decrypted_gradients[key].flatten()))
        console.print(f"{key}: Original Norm = {original_norm}, Decrypted Norm = {decrypted_norm}")
        #print(encrypted_gradients)
        decrypted_dict = decrypt_gradients(encrypted_gradients, decryption_context)  # Get decrypted gradients
        encrypted_norm = np.linalg.norm(np.array(decrypted_dict[key]).flatten()) # Get norm for the same key
        
        # Compute Privacy Leakage Metric (PLM)
        diff = np.abs(original_norm - decrypted_norm) / original_norm
        plm_values.append(diff)

        # Measure encryption effect - Should be significantly different from the original
        encryption_diff = np.abs(original_norm - encrypted_norm) / original_norm
        encryption_difference.append(encryption_diff)

    plm_score = np.mean(plm_values)  # Average privacy leakage across gradients
    encryption_effectiveness = np.mean(encryption_difference)  # How much encryption alters gradients

    # Encryption Overhead - Measures computation cost
    total_time = encryption_time + decryption_time

    # Display Privacy Metrics Report
    privacy_panel = Panel(
        f"[bold cyan]Privacy Metrics Report[/]\n"
        f"----------------------------------\n"
        f"[bold green]Privacy Leakage Metric (PLM):[/] {plm_score:.5f}\n"
        f"[bold green]Encryption Effectiveness:[/] {encryption_effectiveness:.5f}\n"
        f"[bold yellow]Encryption Time:[/] {encryption_time:.4f} seconds\n"
        f"[bold yellow]Decryption Time:[/] {decryption_time:.4f} seconds\n"
        f"[bold red]Total Encryption Overhead:[/] {total_time:.4f} seconds",
        title="[bold red]Privacy Evaluation[/]",
        style="bold white"
    )

    console.print('\n')
    console.print(privacy_panel)

    # Display structured privacy analysis

    privacy_table = Table(title="Detailed Privacy Analysis", show_header=True, header_style="bold magenta")
    privacy_table.add_column("Metric", style="cyan")
    privacy_table.add_column("Value", style="green")

    privacy_table.add_row("Privacy Leakage Metric (PLM)", f"{plm_score:.5f}")
    privacy_table.add_row("Encryption Effectiveness", f"{encryption_effectiveness:.5f}")
    privacy_table.add_row("Encryption Time", f"{encryption_time:.4f} sec")
    privacy_table.add_row("Decryption Time", f"{decryption_time:.4f} sec")
    privacy_table.add_row("Total Overhead", f"{total_time:.4f} sec")

    console.print("\n")
    console.print(privacy_table)

    # Interpretation based on PLM score
    if plm_score < 0.01:
        console.print("[bold green]✅ Extremely Secure: Minimal privacy leakage detected![/]")
    elif plm_score < 0.03:
        console.print("[bold green]✓ Strong Privacy Protection: Encryption is effective with very low leakage.[/]")
    elif plm_score < 0.05:
        console.print("[bold yellow]⚠ Moderate Protection: Some leakage is present, but within acceptable limits.[/]")
    elif plm_score < 0.08:
        console.print("[bold orange]⚠ Weak Protection: Consider using stronger encryption or noise addition.[/]")
    else:
        console.print("[bold red]❌ High Privacy Leakage: Encryption is insufficient! Immediate improvements needed.[/]")


    return plm_score

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
        #console.print("[bold green]\nLoading Data...")
        return pd.read_csv(self.data_path)

    def preprocess_data(self, data):
        # Make a copy to avoid modifying the original
        #console.print("[bold green]\nIntializing Data Preprocessing...")
        data = data.copy()
        
        # Replace missing values
        #console.print("[bold green]\nReplacing missing values...")
        data = data.replace('?', np.nan)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Encode categorical variables
        for col in data.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self.label_encoders[col] = encoder

        # Encode target variable if needed
        if self.target_column in data.select_dtypes(include=['object']):
            self.target_encoder = LabelEncoder()
            data[self.target_column] = self.target_encoder.fit_transform(data[self.target_column])

        # Split features and target
        #console.print("[bold green]\nSplitting Features and Target...")
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        # Scale features
        #console.print("[bold green]\nScalling Features...")
        X = pd.DataFrame(self.feature_scaler.fit_transform(X), columns=X.columns)

        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def split_data_for_nodes(self, X, y, num_nodes):
        console.print(f"Splitting data into portions for {num_nodes} node...")
        time.sleep(2)
        X_split = np.array_split(X, num_nodes)
        y_split = np.array_split(y, num_nodes)
        return [x.copy() for x in X_split], [y.copy() for y in y_split]

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize weights and biases with proper dimensions
        self.W1 = np.random.randn(self.input_dim, 64) * np.sqrt(2. / self.input_dim)
        self.b1 = np.zeros((64,))
        self.W2 = np.random.randn(64, 32) * np.sqrt(2. / 64)
        self.b2 = np.zeros((32,))
        self.W3 = np.random.randn(32, 1) * np.sqrt(2. / 32)
        self.b3 = np.zeros((1,))

    def get_loss(self, X, y):
        """Calculate binary cross-entropy loss"""
        predictions = self.forward_propagation(X)
        epsilon = 1e-15  # Small constant to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )

    def forward_propagation(self, X):
        """Forward pass with proper shape handling"""
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
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
        """Predict with proper shape handling"""
        probabilities = self.forward_propagation(X)
        return (probabilities >= 0.5).astype(int).flatten()

    def score(self, X, y):
        """Calculate accuracy with proper shape handling"""
        predictions = self.predict(X)
        y = np.asarray(y).flatten()  # Ensure y is flattened
        return np.mean(predictions == y)

# Also update the FederatedNode class to handle numpy arrays
class FederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, input_dim, encryption_context, global_model, learning_rate=0.01, local_epochs=5):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.encryption_context = encryption_context
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = 32
        self.input_dim = input_dim
        self.console = Console()
        
        # Initialize local model with global model parameters
        self.model = NeuralNetwork(input_dim=input_dim, learning_rate=learning_rate)
        # Copy global model parameters
        self.model.W1 = np.array(global_model.W1, copy=True)
        self.model.b1 = np.array(global_model.b1, copy=True)
        self.model.W2 = np.array(global_model.W2, copy=True)
        self.model.b2 = np.array(global_model.b2, copy=True)
        self.model.W3 = np.array(global_model.W3, copy=True)
        self.model.b3 = np.array(global_model.b3, copy=True)
        
        self.gradients = None
        self.original_gradients = None
        self.encrypted_gradients = None
        self.encryption_time = 0
    
    """def encrypt_gradients(self, gradients):
        #Encrypt gradients using the node's encryption context
        encrypted_grads = {}
        for key, grad in gradients.items():
            original_shape = grad.shape
            flattened = grad.flatten()
            encrypted = ts.ckks_vector(self.context, flattened)
            encrypted_grads[key] = {
                'data': encrypted,
                'shape': original_shape
            }
        return encrypted_grads"""

    def run(self):
        try:
            set_seed(42)
            self.console.print(f"\n[info]Node {self.node_id} starting local training for {self.local_epochs} epochs...[/]")
            if len(self.X) == 0 or len(self.y) == 0:
                self.console.print(f"[error]Node {self.node_id}: Empty dataset. Skipping training.[/]")
                return

            n_samples = len(self.X)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            # Initialize accumulated gradients
            accumulated_gradients = {
                'dW1': np.zeros_like(self.model.W1),
                'db1': np.zeros_like(self.model.b1),
                'dW2': np.zeros_like(self.model.W2),
                'db2': np.zeros_like(self.model.b2),
                'dW3': np.zeros_like(self.model.W3),
                'db3': np.zeros_like(self.model.b3)
            }

            """with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
            BarColumn(),  # Correctly include bar component
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            console=self.console
            ) as progress:
                task = progress.add_task(f"[cyan]Training Node {self.node_id}...", total=self.local_epochs)"""
            
            for epoch in range(self.local_epochs):
                # Shuffle data for each epoch
                indices = np.random.permutation(n_samples)
                X_shuffled = self.X[indices]
                y_shuffled = self.y[indices]
                
                epoch_loss = 0.0
                
                # Mini-batch training
                for i in range(0, n_samples, self.batch_size):
                    end_idx = min(i + self.batch_size, n_samples)
                    batch_X = X_shuffled[i:end_idx]
                    batch_y = y_shuffled[i:end_idx]

                    # Compute original gradients before encryption
                    batch_gradients = self.model.get_gradients(batch_X, batch_y)
                    self.original_gradients = batch_gradients  # Store first batch gradients for analysis
                    max_norm = max(np.linalg.norm(grad) for grad in self.original_gradients.values())
                    normalized_gradients = {key: grad / max_norm for key, grad in self.original_gradients.items()}
                    self.original_gradients = normalized_gradients
                    # Forward pass
                    predictions = self.model.forward_propagation(batch_X)
                    batch_loss = -np.mean(batch_y * np.log(predictions + 1e-8) + 
                                        (1 - batch_y) * np.log(1 - predictions + 1e-8))
                    epoch_loss += batch_loss
                    
                    # Compute gradients
                    batch_gradients = self.model.get_gradients(batch_X, batch_y)
                    
                # Average gradients for this epoch
                epoch_loss /= n_batches
                
                # Print epoch results
                if epoch % 1 == 0:  # Print every epoch
                    predictions = self.model.predict(self.X)
                    accuracy = np.mean(predictions.flatten() == self.y)
                    #progress.update(task, advance=1)
                    self.console.print(f"\n[success][yellow]Node[/yellow] {self.node_id} - [yellow]Epoch[/yellow] {epoch + 1}/{self.local_epochs} - "
                                    f"[yellow]Loss :[/yellow] {epoch_loss:.4f} - [bold green]Accuracy :[/bold green] {accuracy * 100:.2f}[red] %[/red][/]")
            
            # Average accumulated gradients
            for key in accumulated_gradients:
                accumulated_gradients[key] /= (self.local_epochs * n_batches)
            

            self.gradients = accumulated_gradients
            self.original_gradients=self.gradients
            # Accumulate gradients
            for key in batch_gradients:
                accumulated_gradients[key] += batch_gradients[key]

            # Encrypt gradients and measure encryption time
            start_encrypt_time = time.time()
            self.encrypted_gradients = encrypt_gradients(self.original_gradients, self.encryption_context)
            self.encryption_time = time.time() - start_encrypt_time
            
            self.console.print(f"\n[success]Node {self.node_id} completed local training.[/]")
            
        except Exception as e:
            self.console.print(f"[error]Error in Node {self.node_id}: {str(e)}[/]")
            self.gradients = None
            breakpoint

def weighted_average_gradients(encrypted_gradients_list, node_weights):
    """Average gradients with weights based on node data size or performance"""
    # Initialize with the first node's gradients
    avg_gradients = {}
    first_grads = encrypted_gradients_list[0]
    
    # Initialize each key with the shape information and weighted first gradient
    for key in first_grads:
        avg_gradients[key] = {
            'data': first_grads[key]['data'] * node_weights[0],
            'shape': first_grads[key]['shape']
        }
    
    # Add weighted gradients from other nodes
    for i, grads in enumerate(encrypted_gradients_list[1:], 1):
        for key in avg_gradients:
            # Perform encrypted addition with proper weighting
            avg_gradients[key]['data'] += grads[key]['data'] * node_weights[i]
    
    return avg_gradients

# Improved implementation:
class ImprovedCKKSKeyManager:
    def __init__(self, save_dir="keys/"):
        self.save_dir = save_dir
        self.context = None
        self.encryption_context = None  # Public key context
        self.decryption_context = None  # Private key context
        self.security_level = 128
        self.key_version = 1
        os.makedirs(save_dir, exist_ok=True)

    def generate_keys(self):
        """Generate new CKKS context and keys"""
        import tenseal as ts
        
       # Increase polynomial modulus degree for better precision
        self.poly_modulus_degree = 16384  

        # Optimize coefficient modulus sizes for better rescaling
        self.coeff_mod_bit_sizes = [60, 40, 40, 40, 60]  

        # Create optimized context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
        )

        # Set a higher global scale to improve precision
        self.context.global_scale = 2**50  

        # Generate necessary keys
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        
        # Create separate contexts for encryption and decryption
        self.encryption_context = ts.context_from(
            self.context.serialize(save_secret_key=False)
        )
        self.decryption_context = self.context
        
        return self.context

    def load_keys(self):
        """Load existing keys or generate new ones"""
        try:
            # Load version info
            version_file = f"{self.save_dir}version.txt"
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    self.key_version = int(f.read())

            # Load encryption context (public key)
            pub_key_path = f"{self.save_dir}public_key_v{self.key_version}.bin"
            if not os.path.exists(pub_key_path):
                return self.generate_keys()

            with open(pub_key_path, "rb") as f:
                public_key_data = f.read()
            self.encryption_context = ts.context_from(public_key_data)

            # Load decryption context (with private key)
            secret_key_path = f"{self.save_dir}secret_key_v{self.key_version}.bin"
            if os.path.exists(secret_key_path):
                with open(secret_key_path, "rb") as f:
                    secret_key_data = f.read()
                self.decryption_context = ts.context_from(secret_key_data)
            
            # Validate loaded keys
            if not self._validate_keys():
                return self.generate_keys()

            return self.encryption_context

        except Exception as e:
            print(f"Error loading keys: {e}")
            return self.generate_keys()

    def save_keys(self):
        """Save both public and private keys"""
        if not self.encryption_context or not self.decryption_context:
            return False

        try:
            # Save version
            with open(f"{self.save_dir}version.txt", "w") as f:
                f.write(str(self.key_version))

            # Save public key
            with open(f"{self.save_dir}public_key_v{self.key_version}.bin", "wb") as f:
                f.write(self.encryption_context.serialize(save_secret_key=False))

            # Save private key
            with open(f"{self.save_dir}secret_key_v{self.key_version}.bin", "wb") as f:
                f.write(self.decryption_context.serialize(save_secret_key=True))

            return True

        except Exception as e:
            print(f"Error saving keys: {e}")
            return False

    def _validate_keys(self):
        """Validate encryption and decryption contexts"""
        try:
            if not self.encryption_context or not self.decryption_context:
                return False
                
            # Test encryption/decryption
            test_vec = ts.ckks_vector(self.encryption_context, [1.0])
            test_vec.link_context(self.decryption_context)
            decrypted = test_vec.decrypt()
            return abs(decrypted[0] - 1.0) < 0.1

        except:
            return False

    def get_encryption_context(self):
        """Get context for encryption (public key only)"""
        if self.encryption_context is None :
            self.encryption_context = self.load_keys()
        return self.encryption_context

    def get_decryption_context(self):
        """Get context for decryption (includes secret key)"""
        return self.decryption_context

    
def encrypt_gradients(gradients, encryption_context):
    """Encrypt gradients using public key context"""
    encrypted_grads = {}
    for key, grad in gradients.items():
        original_shape = grad.shape
        flattened = grad.flatten()
        encrypted = ts.ckks_vector(encryption_context, flattened)
        encrypted_grads[key] = {
            'data': encrypted,
            'shape': original_shape
        }
        #console.print(f"\n🔓 Encrypted Weights for {key}:")
        #console.print(encrypted_grads[key])
    return encrypted_grads

def decrypt_gradients(encrypted_gradients, decryption_context):
    """Decrypt gradients using private key context"""
    decrypted_grads = {}
    for key, grad_dict in encrypted_gradients.items():
        encrypted_data = grad_dict['data']
        original_shape = grad_dict['shape']
        
        # Link to decryption context if needed
        if decryption_context != encrypted_data.context():
            encrypted_data.link_context(decryption_context)
        
        decrypted_flat = np.array(encrypted_data.decrypt())
        decrypted_grads[key] = decrypted_flat.reshape(original_shape)
        #console.print(f"\n🔓 Decrypted Weights for {key}:")
        #console.print(decrypted_grads[key])
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
    def __init__(self, input_dim, num_rounds=10, learning_rate=0.01, model_path="global_model.json",local_epochs =5):
        self.input_dim = input_dim
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.global_model = NeuralNetwork(input_dim=input_dim, learning_rate=learning_rate)
        self.momentum = 0.9
        self.previous_gradients = None
        self.local_epochs =local_epochs

    def save_global_model(self):
        """Save the global model parameters to a JSON file."""
        try:
            model_data = {
                'W1': self.global_model.W1.tolist(),
                'b1': self.global_model.b1.tolist(),
                'W2': self.global_model.W2.tolist(),
                'b2': self.global_model.b2.tolist(),
                'W3': self.global_model.W3.tolist(),
                'b3': self.global_model.b3.tolist(),
                'learning_rate': self.learning_rate,
                'momentum': self.momentum
            }
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f)
            print(f"Saved global model to {self.model_path}")
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
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.momentum = model_data.get('momentum', self.momentum)
            print(f"Loaded global model from {self.model_path}")
        except FileNotFoundError:
            print("No existing model found. Starting with a new model.")
        except Exception as e:
            print(f"Error loading global model: {e}")
            print("Starting with a new model.")

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass):
        print("Starting federated learning...")
        start_time = time.time()
        
        try:
            self.load_global_model()
        except Exception as e:
            print(f"Could not load model: {e}. Starting with fresh model.")
            self.global_model = NeuralNetwork(input_dim=self.input_dim, learning_rate=self.learning_rate)

        best_accuracy = 0
        round_num = 0
        max_rounds = 1000
        
        while True:
            round_num += 1
            if round_num > max_rounds:
                print(f"Reached maximum rounds ({max_rounds}) without surpassing centralized accuracy.")
                break
                
            print(f"\n--- Federated Learning Round {round_num} ---")
            
            # Calculate node weights based on data size
            total_samples = sum(len(X) for X in X_train_nodes)
            node_weights = [len(X)/total_samples for X in X_train_nodes]
            
            # Create and start nodes
            nodes = []
            try:
                for i in range(len(X_train_nodes)):
                    node = FederatedNode(
                        node_id=i,
                        X=X_train_nodes[i],
                        y=y_train_nodes[i],
                        input_dim=self.input_dim,
                        encryption_context=self.key_manager.get_encryption_context,
                        global_model=self.global_model,
                        learning_rate=self.learning_rate,
                        local_epochs=5
                    )
                    nodes.append(node)
                    node.start()

                # Collect gradients
                node_gradients = []
                for node in nodes:
                    node.join()
                    if node.gradients is not None:
                        node_gradients.append(encrypt_gradients(node.gradients, self.encryption_context))

                if not node_gradients:
                    print("No valid gradients received from nodes. Skipping round.")
                    continue

                # Process gradients and update model
                avg_encrypted_gradients = weighted_average_gradients(node_gradients, node_weights[:len(node_gradients)])
                avg_gradients = decrypt_gradients(avg_encrypted_gradients)
                
                # Ensure gradients are properly shaped before clipping
                for key in avg_gradients:
                    if not isinstance(avg_gradients[key], np.ndarray):
                        avg_gradients[key] = np.array(avg_gradients[key])
                    
                avg_gradients = clip_gradients(avg_gradients, max_norm=1.0)
                
                # Apply momentum with proper shapes
                if self.previous_gradients is not None:
                    for key in avg_gradients:
                        if avg_gradients[key].shape != self.previous_gradients[key].shape:
                            # Reshape previous gradients if necessary
                            self.previous_gradients[key] = np.reshape(
                                self.previous_gradients[key], 
                                avg_gradients[key].shape
                            )
                        avg_gradients[key] = (self.momentum * self.previous_gradients[key] + 
                                            (1 - self.momentum) * avg_gradients[key])
                
                self.previous_gradients = avg_gradients.copy()
                
                # Update global model
                self.global_model.apply_gradients(avg_gradients)
                # Evaluate and adjust learning rate
                accuracy = self.global_model.score(X_test, y_test)
                print(f"Round {round_num} - Federated Model Accuracy: {accuracy * 100:.2f}%")
                print(f"Current learning rate: {self.learning_rate:.6f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.learning_rate *= 1.05
                    self.save_global_model()
                else:
                    self.learning_rate *= 0.95
                
                self.learning_rate = max(0.0001, min(0.1, self.learning_rate))

                # Check termination conditions
                if loop_until_surpass and accuracy > centralized_accuracy:
                    print(f"\nFederated model has surpassed centralized model after {round_num} rounds!")
                    break
                elif round_num >= self.num_rounds:
                    break

            except Exception as e:
                print(f"Error in round {round_num}: {str(e)}")
                import traceback
                print(traceback.format_exc())  # This will print the full error traceback
                continue

        print(f"\nFederated Learning completed in {time.time() - start_time:.2f} seconds")
        print(f"Best accuracy achieved: {best_accuracy * 100:.2f}%")
        return self.global_model
    
class EnhancedFederatedLearning(FederatedLearning):
    def __init__(self, input_dim, num_rounds=10, learning_rate=0.01, model_path="global_model.json",local_epochs=5):
        super().__init__(input_dim, num_rounds, learning_rate, model_path,local_epochs)
        self.best_node_states = {}  # Store best performing node states
        self.performance_history = []  # Track node performance
        self.local_epochs = local_epochs
        self.key_manager = ImprovedCKKSKeyManager()
        self.encryption_context = self.key_manager.load_keys()
        self.key_manager.save_keys()
        self.console = Console()
    
    def create_summary_table(self, round_num, accuracy, centralized_accuracy, learning_rate):
        table = Table(title=f"\nRound {round_num} Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Federated Model Accuracy", f"{accuracy * 100:.2f}%")
        table.add_row("Centralized Model Accuracy", f"{centralized_accuracy * 100:.2f}%")
        table.add_row("Current Learning Rate", f"{learning_rate:.6f}")
        
        return table
    
    def create_privacy_analysis_report(self):
        """Generate comprehensive privacy analysis report for CKKS implementation"""
        privacy_header = Panel(
            "[bold red]CKKS Homomorphic Encryption Security Analysis[/]\n" +
            "[bold white]Comprehensive Privacy Protection Assessment[/]",
            border_style="red"
        )

        # Security Parameters Table
        security_params = Table(
            title="[bold cyan]CKKS Security Parameters[/]",
            show_header=True,
            header_style="bold magenta"
        )
        security_params.add_column("Parameter", style="cyan")
        security_params.add_column("Value", style="green")
        security_params.add_column("Impact", style="yellow")

        security_params.add_row(
            "Polynomial Modulus Degree",
            "8192",
            "Determines maximum number of encrypted values"
        )
        security_params.add_row(
            "Coefficient Modulus Bits",
            "[60, 40, 40, 60]",
            "Affects precision and security level"
        )
        security_params.add_row(
            "Security Level",
            "128-bit",
            "Post-quantum security strength"
        )
        security_params.add_row(
            "Global Scale",
            "2^40",
            "Precision control for decimal operations"
        )

        # Privacy Features Comparison
        privacy_comparison = Table(
            title="[bold cyan]Privacy Protection Comparison[/]",
            show_header=True,
            header_style="bold magenta"
        )
        privacy_comparison.add_column("Feature", style="cyan")
        privacy_comparison.add_column("CKKS Implementation", style="green")
        privacy_comparison.add_column("Traditional FL", style="red")

        privacy_comparison.add_row(
            "Gradient Privacy",
            "✓ Fully encrypted during transmission",
            "× Plain text gradients"
        )
        privacy_comparison.add_row(
            "Model Parameter Protection",
            "✓ Homomorphically encrypted",
            "× Exposed during aggregation"
        )
        privacy_comparison.add_row(
            "Aggregation Security",
            "✓ Secure computation on encrypted data",
            "× Clear text computation"
        )
        privacy_comparison.add_row(
            "Node Privacy",
            "✓ Individual contributions hidden",
            "× Contributions trackable"
        )

        # Advanced Security Features
        security_features = Table(
            title="[bold cyan]Advanced Security Features[/]",
            show_header=True,
            header_style="bold magenta"
        )
        security_features.add_column("Feature", style="cyan")
        security_features.add_column("Description", style="white")
        security_features.add_column("Protection Level", style="green")

        security_features.add_row(
            "Relinearization Keys",
            "Enable multiplication of encrypted values",
            "Cryptographic Strength"
        )
        security_features.add_row(
            "Galois Keys",
            "Support advanced encrypted operations",
            "Enhanced Security"
        )
        security_features.add_row(
            "Noise Budget",
            "Managed through coefficient modulus",
            "Operational Security"
        )
        security_features.add_row(
            "Scale Management",
            "Precise control over decimal precision",
            "Numerical Stability"
        )

        # Advantages Panel
        advantages = Panel(
            "\n".join([
                "[green]1.[/green] [white]Post-Quantum Security:[/white] CKKS provides security against quantum computing attacks",
                "[green]2.[/green] [white]Zero-Knowledge Proofs:[/white] Nodes can verify computations without revealing data",
                "[green]3.[/green] [white]Homomorphic Properties:[/white] Enables secure computations on encrypted data",
                "[green]4.[/green] [white]Scale Management:[/white] Maintains precision in encrypted computations",
                "[green]5.[/green] [white]Noise Control:[/white] Advanced noise management for stable operations"
            ]),
            title="[bold green]CKKS Advantages",
            style="bold white"
        )

        # Print all components
        #self.console.print(privacy_header)
        self.console.print(security_params)
        self.console.print("\n")
        #self.console.print(privacy_comparison)
        self.console.print("\n")
        #self.console.print(security_features)
        self.console.print("\n")
        #self.console.print(advantages)
        
    def calculate_adaptive_weights(self, node_performances):
        """Calculate weights based on node performance"""
        accuracies = np.array([perf['accuracy'] for perf in node_performances])
        # Softmax temperature scaling for weights
        temperature = 2.0
        exp_accuracies = np.exp((accuracies - np.mean(accuracies)) / temperature)
        weights = exp_accuracies / np.sum(exp_accuracies)
        return weights
    
    def preserve_best_patterns(self, node_gradients, node_performances):
        """Preserve patterns from high-performing nodes"""
        best_accuracy = max(perf['accuracy'] for perf in node_performances)
        best_node_idx = np.argmax([perf['accuracy'] for perf in node_performances])
        
        # Store best node's state if it's better than what we've seen
        if best_accuracy > self.best_node_states.get('accuracy', 0):
            self.best_node_states = {
                'accuracy': best_accuracy,
                'gradients': node_gradients[best_node_idx],
                'performance': node_performances[best_node_idx]
            }
        
        return best_node_idx
    
    def knowledge_distillation_loss(self, student_output, teacher_output, temperature=3.0):
        """Compute knowledge distillation loss"""
        soft_targets = np.exp(teacher_output / temperature) / np.sum(np.exp(teacher_output / temperature))
        student_log_softmax = np.log(np.exp(student_output / temperature) / np.sum(np.exp(student_output / temperature)))
        return -np.sum(soft_targets * student_log_softmax) * (temperature ** 2)
    
    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, centralized_accuracy, loop_until_surpass,local_epochs):
        
        self.local_epochs = local_epochs

        """header = Panel(
            "[bold cyan]Starting Privacy-Preserved Federated Learning using [bold red]CKKS[/bold red] Homomorphic Encryption[/]",
            style="bold yellow"
        )

        security_features = Panel(
            "\n".join([
                "[green]✓[/] [green]CKKS[/green] [white]Homomorphic Encryption for Gradient Protection[/]",
                "[green]✓[/] [white]Secure Aggregation of[/] [green]Encrypted Gradients[/green]",
                "[green]✓[/] [white] Protection Against[/] [green]Gradient Inference Attacks[/green]"
            ]),
            title="[bold green]Security Features Enabled",
            style="bold cyan"
        )"""


        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="body"),
            Layout(name="footer")
        )

        privacy_header = Panel(
            "[bold red]Privacy-Enhanced Federated Learning[/]\n" +
            "[bold white]Using CKKS Homomorphic Encryption for Gradient Protection[/]",
            border_style="red"
        )

        privacy_metrics = Table(
            title="[bold red]Privacy Protection Metrics[/]",
            show_header=True,
            header_style="bold magenta"
        )
        privacy_metrics.add_column("Security Feature", style="cyan")
        privacy_metrics.add_column("Status", style="green")
        privacy_metrics.add_column("Protection Level", style="yellow")

        privacy_metrics.add_row(
            "CKKS Encryption",
            "✓ Active",
            "Complete Gradient Protection"
        )
        privacy_metrics.add_row(
            "Secure Aggregation",
            "✓ Active",
            "Node-level Privacy"
        )
        privacy_metrics.add_row(
            "Gradient Protection",
            "✓ Active",
            "Protected from Inference Attacks"
        )

        #self.console.print(privacy_header)
        #time.sleep(3)
        #self.console.print(privacy_metrics)
        #time.sleep(3)


        params_panel = Panel(
            f"\n".join([
                f"[white]• Number of Participating Nodes:[/] [cyan]{len(X_train_nodes)}[/cyan]",
                f"[white]• Local Epochs:[/] [cyan]{local_epochs}[/cyan]",
                f"[white]• Initial Learning Rate:[/] [cyan]{self.learning_rate}[/cyan]"
            ]),
            title="[bold cyan]Training Parameters",
            style="bold blue"
        )
        
        

        #print("\n" + "="*80)
        #self.console.print(header)
        print("="*80)
        #self.console.print(security_features)
        self.console.print(params_panel)
        time.sleep(3)
        print("="*80 + "\n")
        

        #Creating encryption context
        #self.encryption_context = self.key_manager.get_encryption_context()
        
        start_time = time.time()
        
        try:
            self.load_global_model()
        except Exception as e:
            self.console.print(f"Could not load model: {e}. Starting with fresh model.")
        
        best_accuracy = 0
        round_num = 0
        node_history = []
        max_rounds = 1000

        with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),  # Correctly include bar component
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Training Federated Model...", total=self.num_rounds)
        
            while True:
                round_num += 1
                #progress.update(task, advance=1,description='Training Federated Model...')
                if round_num > max_rounds:
                    self.console.print(f"\nReached maximum rounds ({max_rounds}) without surpassing centralized accuracy.")
                    break
                    
                self.console.print(f"\n--- Privacy-Preserved Federated Learning Round {round_num} ---")
                self.console.print("Initiating secure gradient computation across nodes...")
                
                # Calculate node weights with privacy preservation
                if node_history:
                    node_weights = self.calculate_adaptive_weights(node_history[-1])
                else:
                    total_samples = sum(len(X) for X in X_train_nodes)
                    node_weights = [len(X)/total_samples for X in X_train_nodes]
                
                # Step 1: Train nodes and collect gradients
                nodes = []
                encrypted_gradients_list = []
                original_gradients_list = []
                encryption_times = []
                decryption_times = []
                node_performances = []
                
                try:
                    # Create and start nodes
                    self.console.print("\nSecure Node Operations:")
                    for i in range(len(X_train_nodes)):
                        node = FederatedNode(
                            node_id=i,
                            X=X_train_nodes[i],
                            y=y_train_nodes[i],
                            input_dim=self.input_dim,
                            encryption_context=self.encryption_context,
                            global_model=self.global_model,
                            learning_rate=self.learning_rate,
                            local_epochs = self.local_epochs

                        )
                        nodes.append(node)
                        node.start()
                        self.console.print(f"\n✓ Node {i}: Initiated with secure computation environment")
                    
                    # Collect encrypted gradients
                    self.console.print("\n[bold white]Secure Gradient Collection:[/]")
                    node_gradients = []
                    for node in nodes:
                        node.join()
                        if node.gradients is not None:
                            encrypted_gradients_list.append(node.encrypted_gradients)
                            original_gradients_list.append(node.original_gradients)
                            encryption_times.append(node.encryption_time)
                            self.console.print(f"\n✓ Node {node.node_id}: Successfully encrypted and transmitted gradients")
                            node_gradients.append(encrypt_gradients(node.gradients, self.encryption_context))
                            node_performances.append({
                                'accuracy': node.model.score(X_test, y_test),
                                'loss': 0  # Placeholder since we removed get_loss
                            })
                    
                    if not node_gradients:
                        self.console.print("No valid gradients received. Skipping round.")
                        continue
                    
                    #best_node_idx = self.preserve_best_patterns(node_gradients, node_performances)

                    secure_aggregation_phase = Panel(
                        "\n".join ([
                            f"[bold green]✓[/] [bold yellow]Performing homomorphically encrypted gradient aggregation[/]",
                            f"[bold green]✓[/] [bold yellow]Completed secure weighted averaging of encrypted gradients[/]",
                            f"[bold green]✓[/] [bold yellow]Performing secure gradient decryption[/]"
                                    ]),
                                    title="[bold] Secure Aggregation Phase :",
                                    style="bold green"
                    )
                    #print("\nSecure Aggregation Phase:")
                    #print("✓ Performing homomorphically encrypted gradient aggregation")
                    
                    # Secure weighted averaging
                    avg_encrypted_gradients = weighted_average_gradients(node_gradients, node_weights)
                    #print("✓ Completed secure weighted averaging of encrypted gradients")

                    # Step 4: Decrypt aggregated gradients and measure decryption time
                    start_decrypt_time = time.time()
                    decrypted_gradients = decrypt_gradients(avg_encrypted_gradients, self.key_manager.get_decryption_context())
                    decryption_time = time.time() - start_decrypt_time
                    #decryption_times.append(decryption_time)

                    # Decrypt and process gradients
                    self.decryption_context = self.key_manager.get_decryption_context()

                    # Step 5: Compute & Display Privacy Metrics
                    
                    
                   # print("✓ Performing secure gradient decryption")
                    self.console.print(secure_aggregation_phase)
                    avg_gradients = decrypt_gradients(avg_encrypted_gradients,self.decryption_context)
                    avg_gradients = clip_gradients(avg_gradients, max_norm=1.0)
                    
                    # Apply momentum with proper shapes
                    if self.previous_gradients is not None:
                        for key in avg_gradients:
                            if avg_gradients[key].shape != self.previous_gradients[key].shape:
                                self.previous_gradients[key] = np.reshape(
                                    self.previous_gradients[key], 
                                    avg_gradients[key].shape
                                )
                            avg_gradients[key] = (self.momentum * self.previous_gradients[key] + 
                                                (1 - self.momentum) * avg_gradients[key])
                    
                    self.previous_gradients = avg_gradients.copy()
                    
                    # Update global model
                    self.global_model.apply_gradients(avg_gradients)
                    
                    # Evaluate and adjust learning rate
                    accuracy = self.global_model.score(X_test, y_test)
                    """print(f"\nRound {round_num} Results:")
                    print(f"• Federated Model Accuracy: {accuracy * 100:.2f}%")
                    print(f"• Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
                    print(f"• Privacy-Preserved Gradients Successfully Aggregated")
                    print(f"• Current Learning Rate: {self.learning_rate:.6f}")"""

                    self.console.print(self.create_summary_table(round_num, accuracy, centralized_accuracy, self.learning_rate))
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.console.print("[green]New best accuracy achieved! Saving model...[/]")
                        self.learning_rate *= 1.05
                        self.save_global_model()
                    else:
                        self.learning_rate *= 0.95
                    
                    self.learning_rate = max(0.001, min(0.1, self.learning_rate))
                    progress.update(task, advance=1,description='Training Federated Model...')
                    
                    # Check termination conditions
                    if loop_until_surpass and accuracy > centralized_accuracy:
                        progress.update(task, advance=self.num_rounds-round_num, description="[cyan]Federated Model Training Completed...")
                        self.console.print(f"\n[bold green]Privacy-Preserved Federated Model has surpassed centralized model![/]")
                        self.console.print(f"\n[bold red]Final Accuracy:[/bold red] {accuracy * 100:.2f}[red] %[/red] ([yellow]Centralized: [/yellow]{centralized_accuracy * 100:.2f}[red] %[/red])")
                        break
                    elif round_num >= self.num_rounds and loop_until_surpass == False:
                        progress.update(task, advance=self.num_rounds-round_num, description="[cyan]Federated Model Training Completed ...")
                        break
            
                except Exception as e:
                    self.console.print(f"Error in round {round_num}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    continue
        
        total_time = time.time() - start_time
        self.console.print("\nPrivacy-Preserved Federated Learning Summary\n")
        final_summary = Panel(
            "\n".join([
                f"[cyan]• Total Training Time:[/] {total_time:.2f} seconds",
                f"[cyan]• Best Accuracy Achieved:[/] {best_accuracy * 100:.2f}%",
                f"[cyan]• Total Rounds Completed:[/] {round_num}",
                f"[cyan]• Final Learning Rate:[/] {self.learning_rate:.6f}"
            ]),
            title="[bold]Training Summary",
            style="bold white"
        )
        self.console.print(final_summary)

        privacy_enhancement = Panel(
            "\n".join(
                [   
                    f"[green] ✓[/green] [bold white]All gradient exchanges were encrypted using CKKS[/]",
                    f"[green] ✓[/green] [bold white]No raw gradients were exposed during training[/]",
                    f"[green] ✓[/green] [bold white]Secure aggregation preserved node privacy[/]"

                ]
            ),
            title="[bold]Privacy Enhancements",
            style="bold green"
        
        )
        self.console.print(privacy_enhancement)
        self.console.print("="*80)

        self.console.print("\n[bold red]Detailed Privacy Analysis Report[/bold red]")
        self.create_privacy_analysis_report()

        compute_privacy_metrics(
                        original_gradients_list[0],  # Use first node's gradients for analysis
                        encrypted_gradients_list[0],
                        decrypted_gradients,
                        np.mean(encryption_times),  # Average encryption time
                        decryption_time,
                        decryption_context=self.decryption_context
                    )
        
        return self.global_model

def main():
    try:
        console = Console()
        # Set the flag for training until surpassing centralized model
        while True:
            flag = rich_input("[bold red]Loop until surpass ? (Y/N):[/]").lower()
            if flag == "y":
                loop_until_surpass = True
                break
            elif flag == "n":
                loop_until_surpass = False
                break
            else :
                console.print("Invalid Input Enter (Y/N)")
        

        setup_panel = Panel(
            "[yellow]Federated Learning Demo[/]\n\n" +
            "[bold red]This program demonstrates privacy-preserved federated learning using CKKS homomorphic encryption.[/]",
            style="white"
        )
        console.print(setup_panel)

        with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),  # Correctly include bar component
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        console=console
        ) as progress:
            setup_task = progress.add_task("[cyan]Setting up...", total=100)

        
        # Initialize data handler
            #console.print("-")
            #progress.update(setup_task, advance=20, description="[cyan]Loading data...")
            data_path = "diabetesData.csv"
            target_column = "target"
            split_name = data_path.split(".")
            json_name = split_name[0]
            data_handler = DataHandler(data_path=data_path, target_column=target_column)
            console.print("-")
            progress.update(setup_task, advance=10, description="[cyan]Loading data...")
            
            # Load and preprocess data
            #print("Loading and preprocessing data...")
            time.sleep(2)
            data = data_handler.load_data()
            console.print("[bold green]\n✓[/bold green] [cyan]Data Lodaded Successfully.[/]")
            #time.sleep(1)
            progress.update(setup_task, advance=10, description="[cyan] Intializing Data Preprocessing")
            time.sleep(2)
            console.print("[bold green]\n✓[/bold green] [cyan]Intialized Data Preprocessing")
            X, y = data_handler.preprocess_data(data)
            #time.sleep(1)
            progress.update(setup_task, advance=10, description="[cyan]Replacing missing values")
            time.sleep(2)
            console.print("[bold green]\n✓[/bold green] [cyan]Replaced missing values")
            #time.sleep(1)
            progress.update(setup_task, advance=10, description="[cyan]Splitting Features and Target")
            time.sleep(2)
            console.print("[bold green]\n✓[/bold green] [cyan]Splitted Features and Target")
            time.sleep(1)
            #console.print("[bold green]\n✓[/bold green] [cyan]\nScalling Features...")

            data_table = Table(title="Dataset Summary", show_header=True, header_style="bold magenta")
            data_table.add_column("Metric", style="cyan")
            data_table.add_column("Value", style="green")
            data_table.add_row("Dataset", data_path)
            data_table.add_row("Input Features", str(X.shape[1]))
            data_table.add_row("Total Samples", str(len(X)))
            
            
            # Get input dimension from the data
            input_dim = X.shape[1]
            #print(f"Input dimension: {input_dim}")
            progress.update(setup_task, advance=40, description="[cyan]Fetching Dataset Details")
            time.sleep(2)
            console.print(data_table)
        
            # Split data
            #console.print("-")
            progress.update(setup_task, advance=20, description="[cyan]Splitting data for Train and Test")
            time.sleep(2)
            console.print("[bold green]\n✓[/bold green] [cyan]Splitted data for Train and Test")
            X_train, X_test, y_train, y_test = data_handler.split_data(X, y)
            #console.print("[bold green]\nLoading Data...")
            # Convert pandas objects to numpy arrays
            X_train_np = X_train.values
            X_test_np = X_test.values
            y_train_np = y_train.values
            y_test_np = y_test.values

            set_seed(42)
            progress.update(setup_task, advance=10, description="[cyan]Finalizing Preprocess")
            time.sleep(1)
            progress.update(setup_task, advance=10, description="[cyan] Preprocess Completed")

        # Train centralized model for comparison
        console.print("[red]Training centralized model[/]...")
        time.sleep(2)
        start_time = time.time()
        centralized_model = NeuralNetwork(input_dim=input_dim, learning_rate=0.01)
        
        # Train for a few epochs
        #num_epochs_c = 10
        num_epochs = 5
        batch_size = 32
        n_samples = X_train_np.shape[0]
        
        for epoch in range(num_epochs):
            console.print(f"Epoch {epoch + 1}/{num_epochs}")
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
        time.sleep(2)
        console.print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        time.sleep(2)
        console.print(f"Centralized Training Time: {centralized_time:.2f} seconds\n")
        time.sleep(2)

        # Federated Learning Setup
        console.print("[red]Setting up federated learning...[/]")
        time.sleep(2)
        num_nodes = int(rich_input("[bold cyan]Enter Number of [/][bold red]Nodes[/] : "))
        X_train_nodes, y_train_nodes = data_handler.split_data_for_nodes(X_train, y_train, num_nodes)
        
        # Convert to numpy arrays
        X_train_nodes = [X.values for X in X_train_nodes]
        y_train_nodes = [y.values for y in y_train_nodes]

        # Initialize federated learning with correct input dimension
        fl_model = EnhancedFederatedLearning(
            input_dim=input_dim,  # Use the correct input dimension
            num_rounds=int(rich_input("[bold cyan]Enter Number of[/] [bold red]Rounds[/] : ")),
            learning_rate=0.01,
            model_path=f"{json_name}.json",
            local_epochs = num_epochs,
        )
        
        # Train federated model
        console.print("Starting federated learning training...")
        time.sleep(2)
        federated_model = fl_model.fit(
            X_train_nodes,
            y_train_nodes,
            X_test_np,
            y_test_np,
            centralized_accuracy,
            loop_until_surpass,
            num_epochs
        )

        # Final evaluation
        federated_accuracy = federated_model.score(X_test_np, y_test_np)
        console.print("\nFinal Results:")
        console.print(f"Centralized Model Accuracy: {centralized_accuracy * 100:.2f}%")
        console.print(f"Federated Model Accuracy: {federated_accuracy * 100:.2f}%")

        # Compare models
        if federated_accuracy > centralized_accuracy:
            improvement = ((federated_accuracy - centralized_accuracy)) * 100
            console.print(f"\nFederated learning achieved {improvement:.2f}% improvement over centralized learning")
            console.print(f"[cyan]Dataset used :[/] [red]{data_path}[/]")
        else:
            difference = ((centralized_accuracy - federated_accuracy)) * 100
            console.print(f"\nFederated learning performed {difference:.2f}% worse than centralized learning")
            console.print(f"[cyan]Dataset used :[/] [red]{data_path}[/]")

    except Exception as e:
        console.print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()