# Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, debug=False):
    """
    Loads and preprocesses data from a CSV file.

    Args:
        file_path (str): Path to the CSV data file.
        debug (bool): If true, prints debug information on the data. (its for testing)

    Returns:
        X_scaled (array): Scaled feature matrix.
        y (array): the target variable vector.
    """
    data = pd.read_csv(file_path)
    data.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    T = N // 4
    return train_test_split(X, y, train_size=N, test_size=T, stratify=y, random_state=88)

def debug_print_data(X, X_scaled, y):
    """
    use pandas to print all (check no issue in the data)
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    X_df = pd.DataFrame(X)
    X_scaled_df = pd.DataFrame(X_scaled)
    y_df = pd.Series(y)

    print("Original X (features):")
    print(X_df)
    print("\nScaled X (features):")
    print(X_scaled_df)
    print("\nTarget variable y:")
    print(y_df)
    
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
