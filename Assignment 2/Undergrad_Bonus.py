# Team: [Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)]

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Undergrad_KNN_Model import evaluate_model
from data_preprocessing import load_and_preprocess_data, split_data

def grid_search_optimization(clf, param_grid, X_train, y_train):
    """
    Performs grid search optimization for the specified classifier.

    Args:
        clf: Classifier to optimize.
        param_grid: Dictionary of parameters to search.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        best_estimator: The model with the best parameters.
        best_score: The best cross-validation score.
    """
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

def main():
    """
    Main function to load data, train and evaluate models using k-NN and Decision Tree classifiers
    with grid search optimization over different sample sizes.
    """
    # Load and preprocess data from the file
    X, y = load_and_preprocess_data("data.csv")
    results = []

    # Loop over specified training set sizes
    for N in [40, 140, 240, 340, 440]:
        # Split data into training and test sets with specified training size N
        X_train, X_test, y_train, y_test = split_data(X, y, N)

        # k-NN with grid search
        knn = KNeighborsClassifier()
        knn_params = {'n_neighbors': [3, 5, 7, 9]}
        knn_best, knn_best_score = grid_search_optimization(knn, knn_params, X_train, y_train)
        knn_accuracy, knn_precision, knn_recall, knn_f1, knn_roc_auc, knn_time = evaluate_model(knn_best, X_test, y_test)

        # Decision Tree with grid search
        tree = DecisionTreeClassifier()
        tree_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        tree_best, tree_best_score = grid_search_optimization(tree, tree_params, X_train, y_train)
        tree_accuracy, tree_precision, tree_recall, tree_f1, tree_roc_auc, tree_time = evaluate_model(tree_best, X_test, y_test)

        # Store results for both models
        results.append(
            (N, 
             knn_accuracy, knn_precision, knn_recall, knn_f1, knn_roc_auc, knn_time, knn_best_score,
             tree_accuracy, tree_precision, tree_recall, tree_f1, tree_roc_auc, tree_time, tree_best_score)
        )

        # Print the results for each model, displaying N values, metrics, and best scores
        print(f"N={N} | k-NN: Accuracy={knn_accuracy:.2f}, Precision={knn_precision:.2f}, Recall={knn_recall:.2f}, F1={knn_f1:.2f}, ROC AUC={knn_roc_auc:.2f}, Time={knn_time:.4f} sec, Best Score={knn_best_score:.2f}\n" 
              f"N={N} | Decision Tree: Accuracy={tree_accuracy:.2f}, Precision={tree_precision:.2f}, Recall={tree_recall:.2f}, F1={tree_f1:.2f}, ROC AUC={tree_roc_auc:.2f}, Time={tree_time:.4f} sec, Best Score={tree_best_score:.2f}")

    # Save results to output file
    with open("bonus_output.txt", "w") as f:
        for result in results:
            # Write results for each N to the file for both k-NN and Decision Tree models
            f.write(f"N={result[0]} | k-NN: Accuracy={result[1]:.2f}, Precision={result[2]:.2f}, Recall={result[3]:.2f}, F1={result[4]:.2f}, ROC AUC={result[5]:.2f}, Time={result[6]:.4f} sec, Best Score={result[7]:.2f}\n"
                    f"N={result[0]} | Decision Tree: Accuracy={result[8]:.2f}, Precision={result[9]:.2f}, Recall={result[10]:.2f}, F1={result[11]:.2f}, ROC AUC={result[12]:.2f}, Time={result[13]:.4f} sec, Best Score={result[14]:.2f}\n")
            f.write("\n")  # Separate results for readability

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
