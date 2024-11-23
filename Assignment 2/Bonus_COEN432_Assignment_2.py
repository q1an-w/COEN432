# Qian Yi Wang (40211303) --- Philip Carlsson-Coulombe (40208572)
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from Base_COEN432_Assignment_2 import evaluate_model
from data_preprocessing import load_and_preprocess_data, split_data

def grid_search_optimization(clf, param_grid, X_train, y_train):
    """
    Performs grid search optimization for the specified classifier.
    Tracks all validation accuracies for reporting purposes (check output folder)
    """
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search.best_estimator_, grid_search.best_score_, results_df

def plot_validation_accuracies(results_df, param_name, title, filename):
    """
    Plots and saves validation accuracies for grid search parameters.
    
    Args:
        results_df: DataFrame with grid search results.
        param_name: The parameter name to plot against validation accuracy.
        title: Title for the plot.
        filename: Name of the file to save the plot (check output folder) .
    """
    plt.figure(figsize=(10, 6))

    if f'param_{param_name}' not in results_df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in results DataFrame.")

    grouped = results_df.groupby(f'param_{param_name}')['mean_test_score'].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=f"{param_name}")

    plt.title(title)
    plt.xlabel(param_name.capitalize())
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Parameter Values")
    plt.grid(True)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def set_random_seeds(seed=88):
    random.seed(seed)
    np.random.seed(seed)
    check_random_state(seed)

def main():
    set_random_seeds()
    filename = input("Enter the path to the CSV file (ex: data.csv): ")
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

  
    X_train, X_test, y_train, y_test = split_data(X, y, N)
    os.makedirs("./Outputs", exist_ok=True)

    # k-NN Model with Grid Search
    knn = KNeighborsClassifier()
    knn_params = {'n_neighbors':  [x for x in range(1,16)]}
    knn_best, knn_best_score, knn_results = grid_search_optimization(knn, knn_params, X_train, y_train)
    knn_accuracy, knn_precision, knn_recall, knn_f1, knn_time = evaluate_model(knn_best, X_test, y_test)

    # Plot validation accuracies for k-NN
    plot_validation_accuracies(knn_results, 'n_neighbors', "k-NN Validation Accuracies","./Outputs/knn_validation_accuracy.png")
    print("K-NN model plotting done & saved")

    # Decision Tree Model with Grid Search
    tree = DecisionTreeClassifier()
    tree_params = {'max_depth': [None] + [x for x in range(1,51)], 'min_samples_split': [2, 5, 10]}
    tree_best, tree_best_score, tree_results = grid_search_optimization(tree, tree_params, X_train, y_train)
    tree_accuracy, tree_precision, tree_recall, tree_f1, tree_time = evaluate_model(tree_best, X_test, y_test)

    # Plot validation accuracies for Decision Tree
    plot_validation_accuracies(tree_results, 'max_depth', "Decision Tree Validation Accuracies","./Outputs/decision_tree_validation_accuracy.png")
    print("Decision tree plot done & saved")

    
    output_file = "./Outputs/Optimized_KNN_And_Decision_Tree_output.txt"
    with open(output_file, "w") as f:
        f.write(f"N={N} | k-NN: Accuracy={knn_accuracy:.2f}, Precision={knn_precision:.2f}, Recall={knn_recall:.2f}, "
                f"F1={knn_f1:.2f}, Time={knn_time:.4f} sec, Best Score={knn_best_score:.2f}, Best n neighbours={knn_best.get_params()['n_neighbors']}\n")
        f.write(f"N={N} | Decision Tree: Accuracy={tree_accuracy:.2f}, Precision={tree_precision:.2f}, Recall={tree_recall:.2f}, "
                f"F1={tree_f1:.2f}, Time={tree_time:.4f} sec, Best Score={tree_best_score:.2f},Best Tree Depth={tree_best.get_params()['max_depth']}, Best Min Samples Split={tree_best.get_params()['min_samples_split']}\n")

    print(f"Results saved to {output_file} - Check Ouput Folder")

if __name__ == "__main__":
    main()
