# Team: [Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)]
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Undergrad_KNN_Model import evaluate_model
from data_preprocessing import load_and_preprocess_data, split_data

def grid_search_optimization(clf, param_grid, X_train, y_train):
    
    """
    Performs grid search optimization for the specified classifier.

    Args:
        clf: dlassifier to optimize.
        param_grid: dictionary of parameters to search.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        best_estimator: the model with the best parameters.
        best_score: The best cross-validation score.
    """
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

def main():
    filename = input("Enter the path to the CSV file (e.g., 'data.csv'): ")
    try:
        X, y = load_and_preprocess_data(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return main()
    
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

    # k-NN Model with Grid Search
    knn = KNeighborsClassifier()
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn_best, knn_best_score = grid_search_optimization(knn, knn_params, X_train, y_train)
    knn_accuracy, knn_precision, knn_recall, knn_f1, knn_time = evaluate_model(knn_best, X_test, y_test)

    # Decision Tree Model with Grid Search
    tree = DecisionTreeClassifier()
    tree_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    tree_best, tree_best_score = grid_search_optimization(tree, tree_params, X_train, y_train)
    tree_accuracy, tree_precision, tree_recall, tree_f1, tree_time = evaluate_model(tree_best, X_test, y_test)

    # Print results to the console
    print(f"N={N} | k-NN: Accuracy={knn_accuracy:.2f}, Precision={knn_precision:.2f}, Recall={knn_recall:.2f}, "
          f"F1={knn_f1:.2f}, Time={knn_time:.4f} sec, Best Score={knn_best_score:.2f}")
    print(f"N={N} | Decision Tree: Accuracy={tree_accuracy:.2f}, Precision={tree_precision:.2f}, Recall={tree_recall:.2f}, "
          f"F1={tree_f1:.2f}, Time={tree_time:.4f} sec, Best Score={tree_best_score:.2f}")

    # Save results to a file
    output_file = "bonus_output.txt"
    with open(output_file, "w") as f:
        f.write(f"N={N} | k-NN: Accuracy={knn_accuracy:.2f}, Precision={knn_precision:.2f}, Recall={knn_recall:.2f}, "
                f"F1={knn_f1:.2f}, Time={knn_time:.4f} sec, Best Score={knn_best_score:.2f}\n")
        f.write(f"N={N} | Decision Tree: Accuracy={tree_accuracy:.2f}, Precision={tree_precision:.2f}, Recall={tree_recall:.2f}, "
                f"F1={tree_f1:.2f}, Time={tree_time:.4f} sec, Best Score={tree_best_score:.2f}\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
