# main.py

import pandas as pd 
from src.ga_experiment import run_ga
from src.data_preprocessing import prepare_data 
import numpy as np


def main():
    """
    Main entry point for the BIA601 GA Feature Selection project.
    Handles data loading, preprocessing, running the GA, and presenting results.
    """
    print("\n--- Starting Genetic Algorithm Feature Selection Experiment ---")

    # 1. Configuration (GA Algorithm Parameters)
    GA_PARAMS = {
        'pop_size': 50,
        'generations': 20, 
        'crossover_rate': 0.85,
        'mutation_rate': 0.05,
        'alpha': 0.9, 
        'penalty_weight': 0.1
    }
    
    # 2. Data Preparation (Loading and Preprocessing)
    # Important Note: The path "data/har_data.csv" must be adjusted to match your actual data file location
    DATA_FILE_PATH = "data/har_data.csv" # <<< You must verify this path
    
    try:
        X_train_array, X_test_array, y_train, y_test, feature_names, _ = prepare_data(
            source=DATA_FILE_PATH,
            target="Activity",
            verbose=True
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}. Please check the path.")
        return
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    # [Crucial] Convert X_train and X_test to DataFrame
    # The fitness.py function uses .iloc for feature selection
    X_train_df = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_array, columns=feature_names)

    print(f"\nData loaded: Training set size {X_train_df.shape[0]} samples, {X_train_df.shape[1]} features.")

    # 3. Run Genetic Algorithm (GA Execution)
    best_mask, best_fitness, history = run_ga(
        X_train=X_train_df, 
        y_train=y_train, 
        X_test=X_test_df, 
        y_test=y_test,
        params=GA_PARAMS
    )

    # 4. Presentation of Final Results (Display)
    n_selected = np.sum(best_mask)
    selected_feature_names = [feature_names[i] for i, val in enumerate(best_mask) if val]
    
    print("\n--- Final GA Results ---")
    print(f"Best Fitness Score: {best_fitness:.4f}")
    print(f"Number of Selected Features: {n_selected} / {len(feature_names)}")
    # You can print the list of selected features here
    # print(f"Selected Features: {', '.join(selected_feature_names)}")

    # [Note]: Section 6 call will be added here later

if __name__ == "__main__":
    main()