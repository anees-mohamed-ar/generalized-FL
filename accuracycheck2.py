import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Train a Support Vector Machine (SVM) model
    model = SVC(random_state=42)
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
accuracy*=100
print(f'Accuracy of the model: {accuracy:.2f}')
