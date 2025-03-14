import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CustomNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def calculate_accuracy(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Handle categorical variables with one-hot encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    X_preprocessed = preprocessor.fit_transform(X)

    # Convert target variable to numerical (if it's not numerical)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Train the custom neural network model
    model = CustomNNClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example usage
data_path = 'apple_quality.csv'
target_column = 'Quality'
accuracy = calculate_accuracy(data_path, target_column)
print(f'Accuracy of the model: {accuracy*100:.2f}')
