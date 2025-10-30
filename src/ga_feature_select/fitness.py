# src/ga_feature_select/fitness.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def calculate_fitness(chromosome, ga_instance):

    # Identify selected features
    selected_indices = np.where(chromosome == 1)[0]
    n_selected_features = len(selected_indices)

    if n_selected_features == 0:
        return -1.0, 0.0 # High penalty if no features selected

    # Extract data using only selected features from the GA instance
    X_train_selected = ga_instance.X_train[:, selected_indices]
    X_test_selected = ga_instance.X_test[:, selected_indices]
    
    # Train a simple classifier (e.g., Logistic Regression)
    model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
    model.fit(X_train_selected, ga_instance.y_train)

    # Evaluate performance (Accuracy on Test Set)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(ga_instance.y_test, y_pred)

    # Apply the complex fitness equation
    feature_penalty_ratio = n_selected_features / ga_instance.N_FEATURES
    
    # Fitness = (Weight * Accuracy) - (Penalty Weight * Feature Ratio)
    fitness = (ga_instance.ALPHA * accuracy) - (ga_instance.PENALTY_WEIGHT * feature_penalty_ratio)

    return fitness, accuracy
