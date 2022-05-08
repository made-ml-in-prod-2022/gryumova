import os
import numpy as np
import pandas as pd
from logging import Logger

from ml_project.models.model import Model
from ml_project.enities.model_params import ModelsParams
from ml_project.enities.clf_params import ClassifierParams


def test_init_model(class_params: dict, mock_logger: Logger, path: str):
    model_params = ModelsParams(
        output_model_path=os.path.join(path, "test_models/model.pkl"),
        params_path=os.path.join(path, "test_models/params.json"),
        metric_path=os.path.join(path, "test_models/metrix.json"),
        save_path=os.path.join(path, "test_models/prediction.csv")
    )
    model = Model(model_params, ClassifierParams(**class_params), mock_logger)

    assert not model.clf
    assert model.models_params == model_params
    assert model.clf_params == ClassifierParams(**class_params)
    assert model.logger == mock_logger


def test_fit_model(class_params: dict, mock_logger: Logger, path: str):
    model_params = ModelsParams(
        output_model_path=os.path.join(path, "test_models/model.pkl"),
        params_path=os.path.join(path, "test_models/params.json"),
        metric_path=os.path.join(path, "test_models/metrix.json"),
        save_path=os.path.join(path, "test_models/prediction.csv")
    )
    model = Model(model_params, ClassifierParams(**class_params), mock_logger)

    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    Y = np.array([1, 1, 2, 2])

    model.fit(X, Y)
    model.predict([[-0.8, -1]], os.path.join(path, "test_models/prediction.csv"))
    assert 1 == pd.read_csv(os.path.join(path, "test_models/prediction.csv"), index_col=0).iloc[0, 0]
