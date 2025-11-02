# src/baseline_models.py

import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress warnings from scikit-learn for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, float], Any]:
    """
    Trains a specified classification model and evaluates its performance on the test set.
    """
    
    # 1. Initialize the Model
    if model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=500, solver='liblinear', random_state=random_state)
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from LogisticRegression, DecisionTree, RandomForest.")

    # 2. Train the Model
    model.fit(X_train, y_train)

    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)
    
    # Use 'weighted' average for multi-class classification metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    return metrics, model


def run_baseline_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_to_run: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Runs a comparison across multiple baseline models using all available features.
    """
    if models_to_run is None:
        models_to_run = ['LogisticRegression', 'DecisionTree', 'RandomForest']
    
    all_baseline_results = {}
    print("\n--- Starting Baseline Model Comparison (Using ALL features) ---")

    # Iterate through the selected models
    for model_name in models_to_run:
        print(f"-> Training {model_name}...")
        
        # Train and evaluate using ALL features
        metrics, _ = train_and_evaluate_model(
            model_name, X_train, y_train, X_test, y_test
        )
        
        all_baseline_results[model_name] = metrics
        
        print(f"   {model_name} Accuracy: {metrics['accuracy']:.4f}")

    return all_baseline_results
