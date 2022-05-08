from dataclasses import dataclass, field
from .feature_params import FeatureParams
from .model_params import ModelsParams
from .clf_params import ClassifierParams
from .logger_params import LoggerParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    feature_params: FeatureParams = field(default_factory=FeatureParams)
    models_params: ModelsParams = field(default_factory=ModelsParams)
    class_params: ClassifierParams = field(default_factory=ClassifierParams)
    logging: LoggerParams = field(default_factory=LoggerParams)


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParamsSchema:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
