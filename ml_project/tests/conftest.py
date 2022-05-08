import os
import pytest
import pandas as pd
import numpy as np

from ml_project.enities.feature_params import FeatureParams
from .logger import MockLogger


@pytest.fixture()
def input_data_path():
    curdir = os.path.dirname(__file__)
    size = 300

    data = pd.DataFrame({
        'age': np.random.randint(20, 80, size),
        'cp': np.random.randint(0, 4, size),
        'sex': np.random.randint(0, 2, size),
        'trestbps': np.random.randint(100, 200, size),
        'chol': np.random.randint(120, 560, size),
        'fbs': np.random.randint(0, 2, size),
        'restecg': np.random.randint(0, 2, size),
        'thalach': np.random.randint(71, 202, size),
        'exang': np.random.randint(0, 2, size),
        'oldpeak': np.random.randint(0, 7, size),
        'slope': np.random.randint(0, 3, size),
        'ca': np.random.randint(0, 5, size),
        'thal': np.random.randint(1, 4, size),
        'target': np.random.randint(0, 2, size),
    })

    pd.DataFrame.from_dict(data).to_csv(os.path.join(curdir, "train_data_sample.csv"), index=False)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def path():
    currdir = os.path.dirname(__file__)
    return currdir


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def cat_features() -> dict:
    return []


@pytest.fixture()
def num_features() -> dict:
    return ["age", "thalach",
            "oldpeak", "chol",
            "trestbps", "cp",
            "thal", "slope",
            "ca", "restecg",
            "fbs", "exang", "sex"]


@pytest.fixture()
def features_to_drop() -> dict:
    return []


@pytest.fixture()
def all_features() -> dict:
    return {
        "binary_cols": [],
        "categorical_cols": [],
        "numerical_cols": ["age", "thalach",
                           "oldpeak", "chol",
                           "trestbps", "cp",
                           "thal", "slope",
                           "ca", "restecg",
                           "fbs", "exang", "sex"],
        "target_col": "condition"}


@pytest.fixture()
def mock_logger() -> MockLogger:
    return MockLogger("log")


@pytest.fixture()
def feature_params(num_features, target_col) -> FeatureParams:
    return FeatureParams(
        [],
        num_features,
        [],
        target_col,
        missing_categorial="most_frequent",
        missing_numeric="mean"
    )


@pytest.fixture()
def class_params() -> dict:
    return {"type": "LogisticRegression",
            "loss": "log",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 10000}


@pytest.fixture()
def fake_data():
    size = 300

    data = pd.DataFrame({
        'age': np.random.randint(20, 80, size),
        'cp': np.random.randint(0, 4, size),
        'sex': np.random.randint(0, 2, size),
        'trestbps': np.random.randint(100, 200, size),
        'chol': np.random.randint(120, 560, size),
        'fbs': np.random.randint(0, 2, size),
        'restecg': np.random.randint(0, 2, size),
        'thalach': np.random.randint(71, 202, size),
        'exang': np.random.randint(0, 2, size),
        'oldpeak': np.random.randint(0, 7, size),
        'slope': np.random.randint(0, 3, size),
        'ca': np.random.randint(0, 5, size),
        'thal': np.random.randint(1, 4, size),
        'target': np.random.randint(0, 2, size),
    })

    return data
