from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml
from typing import Optional

from .logging_params import LoggerParams


@dataclass()
class AppParams:
    host: str = '0.0.0.0'
    port: int = 8000
    model_path: str = '../models/model.pkl'
    features: list = None
    logging: Optional[LoggerParams] = field(default_factory=LoggerParams)


def read_app_params(path: str) -> AppParams:
    with open(path, 'r') as input_stream:
        app_schema = class_schema(AppParams)()
        app_params = app_schema.load(yaml.safe_load(input_stream))
        return app_params
