# model_training.py
# Team: [Your Name, Your ID(s)]
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data, split_data
import time

def evaluate_model(clf, X_test, y_test):
    # Predict and calculate metrics
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, test_time

def main():
    X, y = load_and_preprocess_data("data.csv")
    results = []
    for N in [40, 140, 240, 340, 440]:
        X_train, X_test, y_train, y_test = split_data(X, y, N)
        
        # Train k-NN model
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        # Evaluate model
        accuracy, precision, recall, f1, test_time = evaluate_model(clf, X_test, y_test)
        
        # Store results
        results.append((N, accuracy, precision, recall, f1, test_time))
        print(f"N={N}, k-NN Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Time={test_time:.4f} sec")
    
    # Save results to output file
    with open("output.txt", "w") as f:
        for result in results:
            f.write(f"N={result[0]}, k-NN Accuracy={result[1]:.2f}, Precision={result[2]:.2f}, Recall={result[3]:.2f}, F1={result[4]:.2f}, Time={result[5]:.4f} sec\n")

if __name__ == "__main__":
    main()
