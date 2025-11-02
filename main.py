# main.py

import pandas as pd 
from src.ga_experiment import run_ga
from src.data_preprocessing import prepare_data 
import numpy as np
from src.baseline_models import run_baseline_comparison # Import Section 6 comparison function
from typing import Dict, Any


def main():
    """
    Main entry point for the BIA601 GA Feature Selection project.
    Handles data loading, preprocessing, running the GA, and presenting comparison results.
    """
    print("\n--- Starting Genetic Algorithm Feature Selection Experiment ---")

    # 1. Configuration (GA Algorithm Parameters)
    GA_PARAMS: Dict[str, Any] = {
        'pop_size': 50,
        'generations': 20, 
        'crossover_rate': 0.85,
        'mutation_rate': 0.05,
        'alpha': 0.9, 
        'penalty_weight': 0.1
    }
    
    # 2. Data Preparation (Loading and Preprocessing)
    # The data preprocessing function has been modified to handle separate train.csv and test.csv files.
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test.csv"
    
    try:
        # Pass both file paths to the updated prepare_data function
        X_train_array, X_test_array, y_train, y_test, feature_names, _ = prepare_data(
            train_file_path=TRAIN_PATH,
            test_file_path=TEST_PATH,
            target="Activity",
            verbose=True
        )
    except FileNotFoundError:
        print(f"Error: Data file not found. Check if '{TRAIN_PATH}' and '{TEST_PATH}' exist.")
        return
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    # Convert processed NumPy arrays back to Pandas DataFrames (required for fitness.py .iloc)
    X_train_df = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_array, columns=feature_names)

    print(f"\nData loaded: Training set size {X_train_df.shape[0]} samples, {X_train_df.shape[1]} features.")

    # 3. Run Genetic Algorithm (Section 2-5 Execution)
    best_mask, best_fitness, history = run_ga(
        X_train=X_train_df, 
        y_train=y_train, 
        X_test=X_test_df, 
        y_test=y_test,
        params=GA_PARAMS
    )

    # 4. Presentation of GA Final Results
    n_selected = np.sum(best_mask)
    selected_feature_names = [feature_names[i] for i, val in enumerate(best_mask) if val]
    
    print("\n--- GA Feature Selection Results (Section 2-5) ---")
    print(f"Best Fitness Score (Weighted): {best_fitness:.4f}")
    
    # Calculate the raw accuracy achieved by the best mask for fair comparison
    _, raw_accuracy = calculate_fitness(
        best_mask, X_train_df, y_train, X_test_df, y_test, 
        ALPHA=1.0, PENALTY_WEIGHT=0.0 # Use standard accuracy calculation
    )
    print(f"Best Mask Raw Model Accuracy (Logistic Regression): {raw_accuracy:.4f}")
    
    print(f"Number of Selected Features: {n_selected} / {len(feature_names)}")
    # print(f"Selected Features: {', '.join(selected_feature_names)}")

    # 5. Run Baseline Model Comparison (Section 6)
    # Baseline models use all features (X_train_df)
    baseline_results = run_baseline_comparison(
        X_train=X_train_df, 
        y_train=y_train, 
        X_test=X_test_df, 
        y_test=y_test,
    )
    
    # 6. Final Summary and Comparison
    print("\n--- Final Project Summary ---")
    print("GA Feature Selection Results:")
    print(f"  - Selected Features Count: {n_selected}")
    print(f"  - Model Accuracy (GA Selected Features): {raw_accuracy:.4f}")
    
    print("\nBaseline Model Results (Using ALL Features):")
    for model_name, metrics in baseline_results.items():
        print(f"  - {model_name} Accuracy: {metrics['accuracy']:.4f}")
    
    # Additional comparison point if needed:
    # Check accuracy of LR with ALL features vs LR with GA features
    if 'LogisticRegression' in baseline_results:
        lr_baseline_acc = baseline_results['LogisticRegression']['accuracy']
        print("\nLogistic Regression Comparison:")
        print(f"  - LR (ALL Features): {lr_baseline_acc:.4f}")
        print(f"  - LR (GA Features):  {raw_accuracy:.4f}")


if __name__ == "__main__":
    # Import calculate_fitness here to avoid circular dependency issues if the files were not structured properly
    try:
        from src.ga_feature_select.fitness import calculate_fitness
    except ImportError:
        print("Error: Could not import calculate_fitness. Ensure ga_feature_select package structure is correct.")
        exit()
        
    main()