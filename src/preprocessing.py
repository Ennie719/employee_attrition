import os
from typing import Iterable, Optional, Union

import pandas as pd
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw_dataset.csv")
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "config", "configs.yaml")


def load_config(path: str) -> dict:
    """Load a YAML config file and return its parsed dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_data(
    path: str = DEFAULT_DATA_PATH,
    drop_target: bool = False,
    drop_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    By default the target column (Attrition) is kept so downstream modeling can use it.
    Set ``drop_target=True`` to discard the target before feature engineering.
    """
    df = pd.read_csv(path)
    columns_to_drop = set(drop_columns or [])
    for column in ("EmployeeCount", "EmployeeNumber", "StandardHours"):
        if column in df.columns:
            columns_to_drop.add(column)
    if drop_target and "Attrition" in df.columns:
        columns_to_drop.add("Attrition")
    if columns_to_drop:
        df = df.drop(columns=[column for column in columns_to_drop if column in df.columns], errors="ignore")
    return df


def check_data(df: Optional[pd.DataFrame] = None, path: Optional[str] = None):
    """Validate that the dataset has no missing values or duplicates.

    Returns 0 when validation succeeds and raises ValueError otherwise.
    """
    if df is None:
        df = load_data(path or DEFAULT_DATA_PATH)
    if df.isnull().values.any() or df.duplicated().any():
        raise ValueError("There are missing or duplicated value.")
    return 0


def splitting_data(
    df: pd.DataFrame,
    target: str = "Attrition",
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Split a DataFrame into train/test arrays for features and target."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    X = df.drop(columns=[target])
    y = df[target]
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_and_prepare(url_or_df: Union[str, pd.DataFrame], target_column: str = "Attrition"):
    """Compatibility wrapper used by the training notebook.

    Accepts either a file path or an already-loaded DataFrame.
    """
    if isinstance(url_or_df, pd.DataFrame):
        df = url_or_df.copy()
    else:
        df = pd.read_csv(url_or_df)
    df = df.drop(
        columns=[column for column in ("EmployeeCount", "EmployeeNumber", "StandardHours") if column in df.columns],
        errors="ignore",
    )
    if target_column in df.columns and df[target_column].dtype == object:
        df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    return df
