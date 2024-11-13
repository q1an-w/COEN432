# grad_requirements.py
# Team: [Your Name, Your ID(s)]
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from model_training import evaluate_model
from data_preprocessing import load_and_preprocess_data, split_data

def grid_search_optimization(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

def main():
    X, y = load_and_preprocess_data("data.csv")
    results = []
    for N in [40, 140, 240, 340, 440]:
        X_train, X_test, y_train, y_test = split_data(X, y, N)

        # k-NN with grid search
        knn = KNeighborsClassifier()
        knn_params = {'n_neighbors': [3, 5, 7, 9]}
        knn_best, knn_best_score = grid_search_optimization(knn, knn_params, X_train, y_train)
        knn_accuracy, knn_precision, knn_recall, knn_f1, knn_time = evaluate_model(knn_best, X_test, y_test)

        # Decision Tree with grid search
        tree = DecisionTreeClassifier()
        tree_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        tree_best, tree_best_score = grid_search_optimization(tree, tree_params, X_train, y_train)
        tree_accuracy, tree_precision, tree_recall, tree_f1, tree_time = evaluate_model(tree_best, X_test, y_test)

        # Save results
        results.append((N, knn_accuracy, knn_precision, knn_recall, knn_f1, knn_time, tree_accuracy, tree_precision, tree_recall, tree_f1, tree_time))
        print(f"N={N}, k-NN Accuracy={knn_accuracy:.2f}, Decision Tree Accuracy={tree_accuracy:.2f}")
    
    # Save results to output file
    with open("bonus_output.txt", "a") as f:
        for result in results:
            f.write(f"N={result[0]}, k-NN: Accuracy={result[1]:.2f}, Decision Tree: Accuracy={result[6]:.2f}\n")

if __name__ == "__main__":
    main()
