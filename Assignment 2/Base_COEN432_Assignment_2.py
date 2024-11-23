# Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data, split_data
import time
import random

def evaluate_model(clf, X_test, y_test):
    """
    Evaluates the performance of a classifier on the test data.

    Args:
        clf: Trained classifier model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        accuracy: The proportion of correct predictions made by the model on the test dataset.
                           This is calculated as the number of correct predictions divided by the total number of predictions.

        precision  : The ratio of true positive predictions to the total number of positive predictions (both true and false).
                           Precision indicates the model's ability to correctly identify positive cases.

        recall : The ratio of true positive predictions to the total number of actual positive instances.
                        Recall measures the model's ability to detect positive cases.

        f1 : The harmonic mean of precision and recall, providing a balance between the two metrics.
                    F1 score is useful when the data is imbalanced and is calculated as: 
                    F1 = 2 * (precision * recall) / (precision + recall).

        test_time : The time taken (in seconds) to make predictions on the test dataset. This metric helps to evaluate the efficiency of the model in terms of inference time.
    """
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return accuracy, precision, recall, f1, test_time

def set_random_seeds(seed=88):
    random.seed(seed)
    np.random.seed(seed)
    check_random_state(seed)

def main():
    set_random_seeds()
    filename = input("Enter the path to the CSV file (e.g., 'data.csv'): ")
    try:
        X, y = load_and_preprocess_data(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    results = []
    
    N_max = int(len(X)//(5/4))
    print(f"Choose a value for N (8 - {N_max})")
    try:
        N = int(input("Enter your choice for N: "))
        if N < 8 or N > N_max:
            raise ValueError(f"N must be between 8 - {N_max}")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return main()
 
    random.seed(42)
    
    X_train, X_test, y_train, y_test = split_data(X, y, N)
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    accuracy, precision, recall, f1, test_time = evaluate_model(clf, X_test, y_test)
    
    results.append((N, accuracy, precision, recall, f1, test_time))
    print(f"N={N} T={N//4}, k-NN Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Time={test_time:.4f} sec")
    

    os.makedirs("./Outputs", exist_ok=True)
    with open("./Outputs/k-NN output.txt", "w") as f:
        for result in results:
            f.write(f"N={result[0]}, k-NN Accuracy={result[1]:.2f}, Precision={result[2]:.2f}, Recall={result[3]:.2f}, F1={result[4]:.2f}, Time={result[5]:.4f} sec\n")

if __name__ == "__main__":
    main()
