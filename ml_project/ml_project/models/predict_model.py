import click
from logging import Logger

from ml_project.data.make_dataset import read_data
from ml_project.features.build_features import FeatureBuilder
from ml_project.enities.predict_pipepline_params import PredictPipelineParams, read_predict_pipeline_params
from .model import Model
from ..utils import make_logger


def predict_pipeline(params: PredictPipelineParams, logger: Logger):
    logger.info(f"start predict pipeline, params {params}")

    data = read_data(params.models_params.input_data_path)
    logger.info(f"data loaded: {data.shape}")

    pipe = FeatureBuilder(data.columns.values, params.feature_params, logger)
    logger.info(f'transform train data: before shape is {data.shape}')
    pipe.transformer.fit(data)
    X = pipe.transformer.transform(data)
    logger.info(f'transform train data: after shape is  {X.shape}')

    model = Model(params.models_params,
                  params.class_params, logger)
    model.predict(X, params.models_params.save_path)

    logger.info(f"prediction saves in {params.models_params.save_path}")


@click.command(name="predict_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    logger = make_logger("ml_project_hw1", params.logging)

    predict_pipeline(params, logger)


if __name__ == "__main__":
    train_pipeline_command()
