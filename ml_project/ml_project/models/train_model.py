import click
from logging import Logger

from ml_project.enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from ml_project.features.build_features import FeatureBuilder
from ml_project.data.make_dataset import read_data
from ml_project.data.make_dataset import split_tain_val_data
from .model import Model
from ..utils import make_logger


def train(params: TrainingPipelineParams, logger: Logger):
    logger.info(f"start train pipeline, params {params}")

    data = read_data(params.model_params.input_data_path)
    logger.info(f"data loaded: {data.shape}")

    train_df, val_df = split_tain_val_data(data, params.splitting_params)
    logger.info(f"train data shape: {train_df.shape}")
    logger.info(f"valid data shape: {val_df.shape}")

    pipe = FeatureBuilder(data.columns.values, params.feature_params, logger)

    logger.info(f'transform train data: before shape is {train_df.shape}')
    pipe.transformer.fit(train_df)
    X_train = pipe.transformer.transform(train_df)
    y_train = pipe.extract_target(train_df, params.feature_params)
    logger.info(f'transform train data: after shape is  {X_train.shape}')

    model = Model(params.model_params, params.class_params, logger)
    model.fit(X_train, y_train)

    logger.info(f'transform valid data: before shape is {val_df.shape}')
    X_val = pipe.transformer.transform(val_df)
    y_val = pipe.extract_target(val_df, params.feature_params)
    logger.info(f'transform train data: after shape is  {X_val.shape}')

    model.evaluate(X_val, y_val, params.model_params.metric_path)


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)

    logger = make_logger("ml_project_hw1", params.logging)
    train(params, logger)


if __name__ == "__main__":
    train_pipeline_command()
