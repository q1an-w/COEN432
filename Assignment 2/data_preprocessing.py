# Team: [Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, debug=False):
    """
    Loads and preprocesses data from a CSV file.

    Args:
        file_path (str): Path to the CSV data file.
        debug (bool): If True, prints debug information on the data.

    Returns:
        X_scaled (array): Scaled feature matrix.
        y (array): Target variable vector.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns like 'id' and 'Unnamed: 32' if they exist in the dataset
    data.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
    
    # Encode the target variable 'diagnosis' where 'M' -> 1 (Malignant) and 'B' -> 0 (Benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features (X) and target variable (y)
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # Handle any NaN values in the features by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If debug mode is enabled, print data details for troubleshooting
    if debug:
        debug_print_data(X, X_scaled, y)
    
    return X_scaled, y

def split_data(X, y, N):
    """
    Splits the dataset into training and test sets with a 4:1 ratio.

    Args:
        X (array): Feature matrix.
        y (array): Target variable vector.
        N (int): Number of samples in the training set.

    Returns:
        tuple: Training and test sets for features and target variable.
    """
    # Calculate the size of the test set as a quarter of the training size
    T = N // 4
    return train_test_split(X, y, train_size=N, test_size=T, stratify=y, random_state=88)

def debug_print_data(X, X_scaled, y):
    """
    Prints the original and scaled feature matrices and the target variable for debugging.

    Args:
        X (array): Original feature matrix.
        X_scaled (array): Scaled feature matrix.
        y (array): Target variable vector.
    """
    import pandas as pd
    import numpy as np

    # Configure pandas to display the full data frame without truncation
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Convert the data arrays into DataFrames for better visualization
    X_df = pd.DataFrame(X)
    X_scaled_df = pd.DataFrame(X_scaled)
    y_df = pd.Series(y)

    # Display the original and scaled feature matrices, as well as the target variable
    print("Original X (features):")
    print(X_df)
    print("\nScaled X (features):")
    print(X_scaled_df)
    print("\nTarget variable y:")
    print(y_df)
    
    # Reset display options to defaults after displaying data
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
