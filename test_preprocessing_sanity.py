import numpy as np
import pandas as pd
from data_preprocessing import prepare_data

def quick_sanity():
    n_samples = 200
    n_features = 20
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_features)
    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(n_features)])
    df["Activity"] = rng.choice(
        ["WALKING", "SITTING", "STANDING", "LAYING", "WALK_UP", "WALK_DOWN"],
        size=n_samples
    )

    # don’t shrink too much, otherwise test set too small
    X_train, X_test, y_train, y_test, feature_names, preproc = prepare_data(
        df, target="Activity", sample_frac=None, verbose=True
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Num features:", len(feature_names))
    assert X_train.shape[1] == len(feature_names)
    print("Sanity check passed.")

if __name__ == "__main__":
    quick_sanity()
