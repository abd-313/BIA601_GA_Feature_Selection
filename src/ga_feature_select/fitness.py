# src/ga_feature_select/fitness.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from typing import Tuple, Union, Any 

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def calculate_fitness(
    chromosome: np.ndarray, 
    X_train: Any, 
    y_train: Any, 
    X_test: Any, 
    y_test: Any,
    ALPHA: float = 0.9, 
    PENALTY_WEIGHT: float = 0.1
) -> Tuple[float, float]:
    """
    Calculates the multi-objective fitness score of a chromosome (feature mask).

    Fitness is a weighted sum of model accuracy and a penalty for the number of features used.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary mask representing the selected features.
    X_train, y_train, X_test, y_test : Any
        Training and testing data (Pandas DataFrame/Series or NumPy arrays).
    ALPHA : float
        Weight for the accuracy component (default: 0.9).
    PENALTY_WEIGHT : float
        Weight for the feature penalty component (default: 0.1).

    Returns
    -------
    Tuple[float, float]
        (Calculated fitness score, Model accuracy)
    """

    # Identify selected features
    selected_indices = np.where(chromosome == 1)[0]
    n_selected_features = len(selected_indices)
    
    # Determine the total number of features (N_FEATURES)
    N_FEATURES = X_train.shape[1] 

    if n_selected_features == 0:
        return -1.0, 0.0 # High penalty if no features selected

    # Extract data using only selected features
    # Handles both Pandas DataFrames (using iloc) and NumPy arrays
    X_train_selected = X_train.iloc[:, selected_indices] if hasattr(X_train, 'iloc') else X_train[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices] if hasattr(X_test, 'iloc') else X_test[:, selected_indices]
    
    # Train a simple classifier (Logistic Regression)
    model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
    model.fit(X_train_selected, y_train)

    # Evaluate performance (Accuracy on Test Set)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    # Apply the complex fitness equation
    feature_penalty_ratio = n_selected_features / N_FEATURES
    
    # Fitness = (Weight * Accuracy) - (Penalty Weight * Feature Ratio)
    fitness = (ALPHA * accuracy) - (PENALTY_WEIGHT * feature_penalty_ratio)

    return fitness, accuracy