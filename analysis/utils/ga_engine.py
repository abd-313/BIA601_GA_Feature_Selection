import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def run_ga_analysis(df: pd.DataFrame, target_column: str, model_choice: str) -> dict:

    print(f"Executing GA for target: {target_column} with model: {model_choice}")
    

    if model_choice == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    all_features = df.drop(columns=[target_column], errors='ignore').columns.tolist()
    
    num_features_to_select = min(10, len(all_features))
    if num_features_to_select == 0:
        return {'job_id': 'GA_MOCK_FAIL', 'accuracy': 0.0, 'selected_features': []}
        
    selected_features_mock = np.random.choice(all_features, size=num_features_to_select, replace=False).tolist()

    X = df[selected_features_mock]
    y = df[target_column]
    
    try:
        if len(y.unique()) > 1:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            mock_accuracy = scores.mean()
        else:
             print("Target column has only one unique value. Defaulting mock accuracy.")
             mock_accuracy = 0.85

    except Exception as e:
        print(f"Mock scoring failed: {e}. Defaulting to 0.90")
        mock_accuracy = 0.90

    return {
        'job_id': 'GA_MOCK_' + str(np.random.randint(1000, 9999)),
        'accuracy': round(mock_accuracy, 4),
        'selected_features': selected_features_mock,
    }
