import os
import pytest
import pandas as pd
from logging import Logger

from ml_project.models.predict_model import predict_pipeline
from ml_project.enities.predict_pipepline_params import PredictPipelineParams
from ml_project.enities.clf_params import ClassifierParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.model_params import ModelsParams


@pytest.fixture()
def model_params(input_data_path, path: str):
    return ModelsParams(
        input_data_path=input_data_path,
        output_model_path=os.path.join(path, "models/model_log.pkl"),
        metric_path="metrics.json",
        params_path="params.json",
        save_path="prediction.scv"
    )


@pytest.fixture()
def predict_params(model_params, class_params: dict, feature_params: FeatureParams):
    return PredictPipelineParams(
        feature_params=feature_params,
        class_params=ClassifierParams(**class_params),
        models_params=model_params
    )


def test_predict(predict_params, mock_logger: Logger):
    predict_pipeline(predict_params, mock_logger)

    assert os.path.exists(predict_params.models_params.save_path)
    assert os.path.exists(predict_params.models_params.output_model_path)
    assert pd.read_csv(predict_params.models_params.save_path).shape[0] > 1
    assert os.path.exists(predict_params.models_params.metric_path)
