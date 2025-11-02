from typing import Tuple, Optional, List, Union
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def prepare_data(
    source: Union[str, os.PathLike, pd.DataFrame],
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_frac: Optional[float] = None,
    save_preprocessor_path: Optional[str] = "data/preprocessor.joblib",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str], Pipeline]:

    # --- load dataframe ---
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv("source")

    # --- decide target column ---
    if target is None:
        if "Activity" in df.columns:
            target = "Activity"
        else:
            target = df.columns[-1]
            if verbose:
                warnings.warn(f"No 'Activity' column found; using last column '{target}' as target.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns.")

    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # --- ensure numeric features (HAR is numeric; coerce if necessary) ---
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]
    if non_numeric:
        warnings.warn(f"Non-numeric columns detected: {non_numeric}. Attempting to coerce to numeric.")
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found after coercion. HAR expects numeric sensor features.")

    X = X[numeric_cols]

    # --- drop constant columns (zero variance) ---
    nunique = X.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    removed_cols = set(X.columns) - set(keep_cols)
    if removed_cols and verbose:
        warnings.warn(f"Dropping constant columns: {sorted(list(removed_cols))}")
    X = X[keep_cols]

    # --- preprocessor: impute then scale ---
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = list(X.columns)  # order corresponds to X_transformed columns

    # --- optional small sample for debugging ---
    if sample_frac is not None and (0 < sample_frac < 1.0):
        df_small = pd.concat([pd.DataFrame(X_transformed, columns=feature_names), pd.Series(y.reset_index(drop=True), name="__target__")], axis=1)
        df_small = df_small.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        y_small = df_small["__target__"]
        X_small = df_small.drop(columns=["__target__"]).values
        X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=test_size, random_state=random_state, stratify=y_small if y_small.nunique() > 1 else None)
    else:
        stratify = y if (pd.Series(y).nunique() > 1 and y.dtype != float) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    # --- optionally persist preprocessor for web reuse ---
    if save_preprocessor_path:
        os.makedirs(os.path.dirname(save_preprocessor_path), exist_ok=True)
        try:
            joblib.dump(preprocessor, save_preprocessor_path)
            if verbose:
                print(f"Saved preprocessor to: {save_preprocessor_path}")
        except Exception as e:
            warnings.warn(f"Failed to save preprocessor to {save_preprocessor_path}: {e}")

    return X_train, X_test, y_train, y_test, feature_names, preprocessor
