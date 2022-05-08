import os
import pytest
import json
from logging import Logger

from ml_project.models.train_model import train
from ml_project.enities.train_pipeline_params import TrainingPipelineParams
from ml_project.enities.clf_params import ClassifierParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.model_params import ModelsParams
from ml_project.enities.split_params import SplittingParams


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
def split_params():
    return SplittingParams(
        val_size=0.1, random_state=1
    )


@pytest.fixture()
def train_params(split_params, model_params, class_params: dict, feature_params: FeatureParams):
    return TrainingPipelineParams(
        splitting_params=split_params,
        feature_params=feature_params,
        class_params=ClassifierParams(**class_params),
        model_params=model_params
    )


def test_train(train_params, mock_logger: Logger):
    train(train_params, mock_logger)

    assert os.path.exists(train_params.model_params.metric_path)
    with open(train_params.model_params.metric_path) as json_file:
        data = json.load(json_file)

    assert os.path.exists(train_params.model_params.output_model_path)
    assert os.path.exists(train_params.model_params.input_data_path)
    assert 0 <= data["f1_score"] <= 1
    assert 0 <= data["accuracy"] <= 1
    assert 0 <= data["roc_auc_score"] <= 1
