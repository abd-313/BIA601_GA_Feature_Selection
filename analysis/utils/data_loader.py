import pandas as pd
import numpy as np

# This is a placeholder function used by views.py for 'repository' data source.
# You will need to implement the logic for loading data from your pre-defined 
# repository (e.g., a file path or database connection) here later.

def load_repository_data() -> pd.DataFrame:
    """
    Core function to load a predefined repository dataset.
    This currently generates mock numeric data for initial testing.
    """
    print("Loading mock repository data...")
    
    # Create mock numeric data (100 rows, 20 features)
    np.random.seed(42)
    num_samples = 100
    num_features = 20
    
    data = {f'Feature_{i}': np.random.rand(num_samples) * 100 for i in range(1, num_features + 1)}
    
    # Create a target column (binary classification)
    data['Activity'] = np.random.randint(0, 2, size=num_samples)
    
    df = pd.DataFrame(data)
    
    print(f"Mock repository data loaded: {df.shape}")
    return df
