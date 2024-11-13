# data_preprocessing.py
# Team: [Your Name, Your ID(s)]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, debug=True):
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

    # Debug print if enabled
    if debug:
        debug_print_data(X, X_scaled, y)
    
    return X_scaled, y

def split_data(X, y, N):
    """Split the data into training and test sets with a 4:1 ratio."""
    T = N // 4
    return train_test_split(X, y, train_size=N, test_size=T, stratify=y, random_state=42)

def debug_print_data(X, X_scaled, y):
    """Prints X, X_scaled, and y for debugging purposes with full output."""
    import pandas as pd
    import numpy as np

    # Set pandas options to display more rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Convert arrays to DataFrames for better display
    X_df = pd.DataFrame(X)
    X_scaled_df = pd.DataFrame(X_scaled)
    y_df = pd.Series(y)

    print("Original X (features):")
    print(X_df)
    print("\nScaled X (features):")
    print(X_scaled_df)
    print("\nTarget variable y:")
    print(y_df)
    
    # Reset options after display
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

