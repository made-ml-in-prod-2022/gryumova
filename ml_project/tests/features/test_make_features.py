from typing import List
from logging import Logger
import pytest

from ml_project.features.build_features import FeatureBuilder
from ml_project.enities.feature_params import FeatureParams


@pytest.fixture
def features() -> List:
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


@pytest.fixture
def feature_params(
    cat_features: List[str],
    features_to_drop: List[str],
    num_features: List[str],
    target_col: str
) -> FeatureParams:
    return FeatureParams(categorical_features=cat_features,
                         numerical_features=num_features,
                         features_to_drop=features_to_drop,
                         target_col=target_col,
                         missing_categorial="most",
                         missing_numeric="mean")


def test_init_features_builder(
    all_features: List, feature_params: FeatureParams, mock_logger: Logger
):
    fbuilder = FeatureBuilder(all_features, feature_params, mock_logger)

    assert fbuilder.features == all_features
    assert fbuilder.params == feature_params
    assert fbuilder.logger == mock_logger


def test_feature_builder_numeric(all_features: List,
                                 feature_params: FeatureParams,
                                 mock_logger: Logger,
                                 fake_data):
    fbuilder = FeatureBuilder(all_features, feature_params, mock_logger)

    pipe = fbuilder.build_numerical_pipeline(fbuilder.params)
    dataset = pipe.fit_transform(fake_data)

    assert dataset.shape[1] == 14
    assert (dataset.mean() <= 1).all()
    assert (dataset.mean() >= -1).all()


def test_feature_builder_extract(all_features: List,
                                 feature_params: FeatureParams,
                                 mock_logger: Logger,
                                 fake_data):
    fbuilder = FeatureBuilder(all_features, feature_params, mock_logger)
    target = fbuilder.extract_target(fake_data, fbuilder.params)

    assert sorted(target.unique().tolist()) == [0, 1]
