from typing import Optional
from dataclasses import dataclass

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .clf_params import ClassifierParams
from .model_params import ModelsParams
from .logger_params import LoggerParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    splitting_params: Optional[SplittingParams] = None
    feature_params: Optional[FeatureParams] = None
    class_params: Optional[ClassifierParams] = None
    model_params: Optional[ModelsParams] = None
    logging: Optional[LoggerParams] = None


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
