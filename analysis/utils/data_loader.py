import pandas as pd
import numpy as np


def load_repository_data() -> pd.DataFrame:

    print("Loading mock repository data...")
    
    np.random.seed(42)
    num_samples = 100
    num_features = 20
    
    data = {f'Feature_{i}': np.random.rand(num_samples) * 100 for i in range(1, num_features + 1)}
    
    data['Activity'] = np.random.randint(0, 2, size=num_samples)
    
    df = pd.DataFrame(data)
    
    print(f"Mock repository data loaded: {df.shape}")
    return df
