# Team: [Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)]
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

        roc_auc : The Area Under the Receiver Operating Characteristic (ROC) Curve, which represents the model's ability to distinguish between classes.
                         A higher ROC AUC value indicates better model performance, with a value of 1.0 representing perfect discrimination.

        test_time : The time taken (in seconds) to make predictions on the test dataset. This metric helps to evaluate the efficiency of the model in terms of inference time.
    """
    # Predict on test data and measure prediction time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc, test_time

def main():
    """
    Main function to load data, train k-NN models on different sample sizes, and evaluate their performance.
    """
    # Load and preprocess data from the file
    X, y = load_and_preprocess_data("data.csv")
    results = []
    
    # Loop over different training set sizes
    for N in [40, 140, 240, 340, 440]:
        # Ensure reproducible random sampling for each run
        random.seed(42)
        
        # Split data into training and test sets with specified training size N
        X_train, X_test, y_train, y_test = split_data(X, y, N)
        
        # Initialize and train k-NN model with k=5 neighbors
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        # Evaluate the model's performance
        accuracy, precision, recall, f1, roc_auc, test_time = evaluate_model(clf, X_test, y_test)
        
        # Store the results for each N
        results.append((N, accuracy, precision, recall, f1, test_time))
        print(f"N={N} T={N//4}, k-NN Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, ROC AUC={roc_auc:.2f}, Time={test_time:.4f} sec")
    
    # Save results to an output text file
    with open("k-NN output.txt", "w") as f:
        for result in results:
            f.write(f"N={result[0]}, k-NN Accuracy={result[1]:.2f}, Precision={result[2]:.2f}, Recall={result[3]:.2f}, F1={result[4]:.2f}, Time={result[5]:.4f} sec\n")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
