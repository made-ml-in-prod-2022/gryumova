import os
import pickle
import pandas as pd
from logging import Logger

from entities.app_params import AppParams


class ServiceCore:
    def __init__(self):
        self.logger = None
        self.model = None
        self.is_init = False

    def init(self, params: AppParams, logger: Logger):
        model_path = os.path.abspath(params.model_path)
        logger.info(f'Loading model from {model_path}')

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.info('   model loaded')

        self.logger = logger
        self.model = model
        self.is_init = True

        self.logger.info('Core initialized successfully')

    def predict(self, data: pd.DataFrame) -> list:
        datapipe = self.model['datapipe']
        classifier = self.model['model']
        data_ft = datapipe.transform(data)
        preds = classifier.predict(data_ft)

        return list(preds)
