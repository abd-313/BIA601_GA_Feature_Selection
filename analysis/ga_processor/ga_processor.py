# analysis/ga_processor/ga_processor.py
# (English comments as requested)

import pandas as pd 
import numpy as np
import os
import warnings
import json
import shutil
import uuid 
from typing import Dict, Any, Tuple
from pathlib import Path
from joblib import dump 
from sklearn.linear_model import LogisticRegression 

# Project Imports - CRITICAL: Ensure these absolute imports work based on your project structure.
# If Django cannot resolve 'src.data_preprocessing', you might need to adjust the Python path
# or place a symlink/init file to expose the 'src' directory correctly to Django's environment.
from src.data_preprocessing import prepare_data 
from src.ga_experiment import run_ga
from src.baseline_models import run_baseline_comparison 
from src.ga_feature_select.fitness import calculate_fitness 

# --- CONFIGURATION & PATHS ---
# Define the root directory for all GA job results, using a relative path within the Django app.
# IMPORTANT: Create the folder 'ga_job_results' inside your 'analysis' app folder manually.
# Example path: /home/dark/BIA601_GA_Feature_Selection/analysis/ga_job_results
GA_JOB_RESULTS_ROOT = Path(__file__).parent.parent / "ga_job_results"
os.makedirs(GA_JOB_RESULTS_ROOT, exist_ok=True) # Ensure the directory exists

# Default parameters (Ideally, these should come from the Django form)
DEFAULT_GA_PARAMS: Dict[str, Any] = {
    'pop_size': 50,         
    'generations': 20,      
    'crossover_rate': 0.85, 
    'mutation_rate': 0.05,  
    'alpha': 0.9,           
    'penalty_weight': 0.1   
}
# --- END CONFIGURATION ---


def process_ga_job(
    input_data: pd.DataFrame, 
    target_column: str, 
    model_choice: str, # Currently unused in core GA logic but saved for context
    job_id: str, 
    ga_params: Dict[str, Any] = DEFAULT_GA_PARAMS
) -> Dict[str, Any]:
    """
    Runs the Genetic Algorithm feature selection on the provided DataFrame.
    Saves all results (model, mask, summary, plot) to a dedicated job folder.
    
    Args:
        input_data (pd.DataFrame): DataFrame loaded directly from the uploaded CSV.
        target_column (str): The name of the target variable from the form.
        model_choice (str): The selected ML model (for reference).
        job_id (str): A unique identifier for this job.
        ga_params (Dict): Hyperparameters for the GA.

    Returns: 
        Dict: A summary dictionary of results including paths, or an error message.
    """
    
    # 1. Setup paths for this specific job
    JOB_DIR = GA_JOB_RESULTS_ROOT / job_id
    os.makedirs(JOB_DIR, exist_ok=True)
    
    BEST_MODEL_PATH = JOB_DIR / f'{job_id}_model.joblib'
    FEATURE_MASK_PATH = JOB_DIR / f'{job_id}_mask.joblib'
    GA_PLOT_PATH = JOB_DIR 

    # --- CRITICAL ADAPTATION: Save DataFrame to a temporary CSV ---
    # Since your `prepare_data` function expects a file path (source: Union[str, ...]), 
    # we temporarily save the uploaded DataFrame to a CSV file within the job directory 
    # so the existing GA logic can consume it without major changes.
    temp_csv_path = JOB_DIR / f'{job_id}_temp_input.csv'
    # Ensure all data is saved, excluding index
    input_data.to_csv(temp_csv_path, index=False)
    
    data_source_path = str(temp_csv_path) # Path to the temp CSV
    
    # 2. Data Preparation
    try:
        # Pass the file path and the dynamic target column name
        X_train_array, X_test_array, y_train, y_test, feature_names, _ = prepare_data(
            source=data_source_path,
            target=target_column,  # Use dynamic target column from the form
            verbose=False,
            save_preprocessor_path=None
        )

    except Exception as e:
        # Cleanup temporary files on failure
        shutil.rmtree(JOB_DIR, ignore_errors=True)
        return {'error': True, 'message': f"Data Preparation Error: {e}"}

    # Convert NumPy arrays back to Pandas DataFrames for GA/baseline consistency
    X_train_df = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_array, columns=feature_names)

    # 3. Run Genetic Algorithm (GA)
    try:
        warnings.filterwarnings("ignore")
        best_mask, best_fitness, _ = run_ga(
            X_train=X_train_df, 
            y_train=y_train, 
            X_test=X_test_df, 
            y_test=y_test,
            params=ga_params,
            save_path=GA_PLOT_PATH # Plot will be saved in JOB_DIR
        )

        # 4. Final Scoring and Model Training (Logic copied from your main.py)
        n_selected = np.sum(best_mask)
        _, raw_accuracy = calculate_fitness(
            best_mask, X_train_df, y_train, X_test_df, y_test, 
            ALPHA=1.0, PENALTY_WEIGHT=0.0 # Pure accuracy calculation
        )
        
        # Train the final Logistic Regression model
        final_model = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42)
        X_train_best = X_train_df.loc[:, best_mask.astype(bool)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(X_train_best, y_train) 
        
        # 5. Run Baseline Models
        baseline_results = run_baseline_comparison(
            X_train=X_train_df, 
            y_train=y_train, 
            X_test=X_test_df, 
            y_test=y_test,
            verbose=False
        )

    except Exception as e:
        shutil.rmtree(JOB_DIR, ignore_errors=True)
        return {'error': True, 'message': f"Genetic Algorithm Execution Error: {e}"}
    
    # 6. Save final outputs (Model, Mask, Summary JSON)
    try:
        # Save model and mask using joblib
        dump(best_mask, FEATURE_MASK_PATH)
        dump(final_model, BEST_MODEL_PATH)
        
        # Convert baseline results summary DataFrame to a dictionary for JSON
        summary_df = pd.DataFrame(baseline_results).T[['accuracy', 'f1_weighted']]
        summary_df.index.name = "Model"
        baseline_markdown = summary_df.to_markdown(numalign="left", stralign="left")
        
        summary_data = {
            'job_id': job_id,
            'dataset_name': f'Custom Upload: {os.path.basename(data_source_path)}',
            'ml_model': model_choice,
            'target_column': target_column,
            'ga_params': ga_params,
            'ga_weighted_fitness': float(best_fitness),
            'ga_model_accuracy': float(raw_accuracy),
            'n_features_selected': int(n_selected),
            'total_features': len(feature_names),
            'baseline_results_markdown': baseline_markdown,
            'plot_path_relative': f'/{job_id}/fitness_evolution.png', # Relative path for Django results view
            'status': 'Completed',
        }
        with open(JOB_DIR / f'{job_id}_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=4)
            
        # Clean up the temporary input file
        os.remove(temp_csv_path)

    except Exception as e:
        shutil.rmtree(JOB_DIR, ignore_errors=True)
        return {'error': True, 'message': f"Error saving job results: {e}"}

    # Return success summary
    return {
        'error': False,
        'job_id': job_id,
        'status': 'Completed',
        'accuracy': float(raw_accuracy),
        'job_dir': str(JOB_DIR),
        'summary_path': str(JOB_DIR / f'{job_id}_summary.json')
    }