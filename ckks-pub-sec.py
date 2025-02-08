class CKKSKeyManager:
    def __init__(self, save_dir="keys/"):
        self.save_dir = save_dir
        self.context = None
        self.public_key = None
        self.secret_key = None
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def generate_keys(self):
        """Generate CKKS context and keys"""
        import tenseal as ts
        
        # Create context with specified parameters
        poly_modulus_degree = 8192
        coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        
        # Set up context parameters
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        
        # Get public and secret keys
        self.public_key = self.context.serialize(save_secret_key=False)
        self.secret_key = self.context.secret_key()
        
        return self.context

    def save_keys(self):
        """Save keys and context to files"""
        if self.context and self.public_key:
            with open(f"{self.save_dir}public_key.bin", "wb") as f:
                f.write(self.public_key)
            
            if self.secret_key:
                with open(f"{self.save_dir}secret_key.bin", "wb") as f:
                    f.write(self.secret_key)

    def load_keys(self):
        """Load keys and context from files"""
        import tenseal as ts
        try:
            # Load public key and create context
            with open(f"{self.save_dir}public_key.bin", "rb") as f:
                self.public_key = f.read()
            self.context = ts.context_from(self.public_key)
            
            # Load secret key if available
            try:
                with open(f"{self.save_dir}secret_key.bin", "rb") as f:
                    self.secret_key = f.read()
                    self.context.load_secret_key(self.secret_key)
            except FileNotFoundError:
                print("No secret key found - operating in public key mode only")
            
            return self.context
        except FileNotFoundError:
            print("No keys found - generating new keys")
            return self.generate_keys()

    def get_encryption_context(self):
        """Get context for encryption (public key only)"""
        import tenseal as ts
        if self.public_key:
            return ts.context_from(self.public_key)
        return None

    def get_decryption_context(self):
        """Get context for decryption (includes secret key)"""
        if self.context and self.secret_key:
            return self.context
        return None

class ModifiedFederatedNode(threading.Thread):
    def __init__(self, node_id, X, y, input_dim, encryption_context, global_model, 
                 learning_rate=0.01, local_epochs=5):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.X = X
        self.y = y
        self.encryption_context = encryption_context
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = 32
        self.input_dim = input_dim
        
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

    def encrypt_gradients(self, gradients):
        """Encrypt gradients using the node's encryption context"""
        encrypted_grads = {}
        for key, grad in gradients.items():
            original_shape = grad.shape
            flattened = grad.flatten()
            encrypted = ts.ckks_vector(self.encryption_context, flattened)
            encrypted_grads[key] = {
                'data': encrypted,
                'shape': original_shape
            }
        return encrypted_grads

class ModifiedEnhancedFederatedLearning(FederatedLearning):
    def __init__(self, input_dim, num_rounds=10, learning_rate=0.01, 
                 model_path="global_model.json", local_epochs=5):
        super().__init__(input_dim, num_rounds, learning_rate, model_path, local_epochs)
        self.key_manager = CKKSKeyManager()
        self.context = self.key_manager.generate_keys()  # Generate initial keys
        self.key_manager.save_keys()  # Save keys for future use

    def fit(self, X_train_nodes, y_train_nodes, X_test, y_test, 
            centralized_accuracy, loop_until_surpass, local_epochs):
        # Get encryption context for nodes
        encryption_context = self.key_manager.get_encryption_context()
        
        # Initialize nodes with encryption context
        nodes = []
        for i in range(len(X_train_nodes)):
            node = ModifiedFederatedNode(
                node_id=i,
                X=X_train_nodes[i],
                y=y_train_nodes[i],
                input_dim=self.input_dim,
                encryption_context=encryption_context,
                global_model=self.global_model,
                learning_rate=self.learning_rate,
                local_epochs=local_epochs
            )
            nodes.append(node)

        # Rest of the training logic remains the same...
        # When decrypting gradients, use the decryption context
        decryption_context = self.key_manager.get_decryption_context()
        # Use decryption_context for gradient aggregation and decryption