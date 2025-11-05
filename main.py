import pandas as pd 
import numpy as np
import os
import warnings
from typing import Dict, Any, Tuple
from pathlib import Path
from joblib import dump, load  # ADDED: for saving the final model/mask
from sklearn.linear_model import LogisticRegression # ADDED: for final model training/saving


from src.data_preprocessing import prepare_data 
from src.ga_experiment import run_ga
from src.baseline_models import run_baseline_comparison 
from src.ga_feature_select.fitness import calculate_fitness 

PROJECT_ROOT = Path(__file__).parent


BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'best_ga_model.joblib')
FEATURE_MASK_PATH = os.path.join(PROJECT_ROOT, 'data', 'best_feature_mask.joblib')


def run_ga_analysis(data_source: str, ga_params: Dict[str, Any]) -> Dict[str, Any]:

    # Data Preparation
    try:
        X_train_array, X_test_array, y_train, y_test, feature_names, _ = prepare_data(
            source=data_source, 
            target="Activity",
            verbose=False,
            save_preprocessor_path=None
        )
    except Exception as e:
        return {'error': True, 'message': f"Data Preparation Error: {e}"}

    # Convert NumPy arrays to Pandas
    X_train_df = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_array, columns=feature_names)

    # Run Genetic Algorithm (GA)
    GA_SAVE_PATH = os.path.join(PROJECT_ROOT, "temp_plots")
    os.makedirs(GA_SAVE_PATH, exist_ok=True) 
    
    best_mask, best_fitness, _ = run_ga(
        X_train=X_train_df, 
        y_train=y_train, 
        X_test=X_test_df, 
        y_test=y_test,
        params=ga_params,
        save_path=GA_SAVE_PATH
    )

    # Final GA Results and Scoring
    n_selected = np.sum(best_mask)

    
    _, raw_accuracy = calculate_fitness(
        best_mask, X_train_df, y_train, X_test_df, y_test, 
        ALPHA=1.0, PENALTY_WEIGHT=0.0 # Pure accuracy calculation
    )
    

    final_model = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42)
    X_train_best = X_train_df.loc[:, best_mask.astype(bool)] # FIXED: Ensure mask is boolean
    
    # Suppress warnings during training 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train_best, y_train) 
    
    # Run Baseline Models
    baseline_results = run_baseline_comparison(
        X_train=X_train_df, 
        y_train=y_train, 
        X_test=X_test_df, 
        y_test=y_test,
        verbose=False 
    )
    
    
    summary_df = pd.DataFrame(baseline_results).T[['accuracy', 'f1_weighted']]
    summary_df.index.name = "Model"
    
    
    baseline_markdown = summary_df.to_markdown(numalign="left", stralign="left")
    
    results = {
        'error': False,
        'n_samples': X_train_df.shape[0] + X_test_df.shape[0], 
        'total_features': len(feature_names),
        'ga_weighted_fitness': float(best_fitness),
        'ga_model_accuracy': float(raw_accuracy),
        'n_features_selected': int(n_selected),
        'lr_baseline_accuracy': baseline_results['LogisticRegression']['accuracy'],
        'baseline_results_markdown': baseline_markdown,
        'plot_path': os.path.join(GA_SAVE_PATH, "fitness_evolution.png"),
        'best_mask': best_mask, # ADDED: return the best mask
        'best_model': final_model, # ADDED: return the final trained model
    }
    
    return results


if __name__ == "__main__":
    # allows running the script with the default path
    warnings.filterwarnings("ignore")
    
    # Default parameters for testing
    DEFAULT_GA_PARAMS: Dict[str, Any] = {
        'pop_size': 50,         
        'generations': 20,      
        'crossover_rate': 0.85, 
        'mutation_rate': 0.05,  
        'alpha': 0.9,           
        'penalty_weight': 0.1   
    }
    
    # Default data path: data/train.csv
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.csv") 

    print("\n--- Starting GA Feature Selection Experiment (TEST MODE) ---")
    final_results = run_ga_analysis(DATA_PATH, DEFAULT_GA_PARAMS)
    
    if not final_results.get('error'):
        # Extract results for saving
        best_mask = final_results['best_mask']
        best_model = final_results['best_model']
        
        # Save the final outputs for Django deployment
        try:
            # Save the feature mask
            dump(best_mask, FEATURE_MASK_PATH)
            
            # Save the best trained model
            dump(best_model, BEST_MODEL_PATH)
            
            print(f"\n======================================")
            print(f"   FINAL OUTPUTS SAVED FOR DJANGO:")
            print(f"   Feature Mask saved to: {FEATURE_MASK_PATH}")
            print(f"   Best GA Model saved to: {BEST_MODEL_PATH}")
            print(f"======================================\n")
            
        except Exception as e:
            print(f"   Error saving final results: {e}")
            
        print("\n--- Final Project Summary ---")
        print(f"GA Model Accuracy: {final_results['ga_model_accuracy']:.4f}")
        print(f"Features Selected: {final_results['n_features_selected']} / {final_results['total_features']}")
        print("\nBaseline Model Results (All Features):")
        print(final_results['baseline_results_markdown'])
    else:
        print(f"Error during execution: {final_results['message']}")