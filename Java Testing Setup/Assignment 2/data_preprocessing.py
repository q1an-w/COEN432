# data_preprocessing.py
# Team: [Your Name, Your ID(s)]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns like 'id' and 'Unnamed: 32' if they exist
    data.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
    
    # Encode 'diagnosis': M -> 1, B -> 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Split data into features and target variable
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # Handle any NaN values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, N):
    """Split the data into training and test sets with a 4:1 ratio."""
    T = N // 4
    return train_test_split(X, y, train_size=N, test_size=T, stratify=y, random_state=42)
