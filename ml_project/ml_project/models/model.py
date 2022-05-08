from typing import Dict, List
import pandas as pd
import numpy as np
import json
import joblib
from logging import Logger

from ..enities.model_params import ModelsParams
from ml_project.enities.clf_params import ClassifierParams

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def get_clf(class_params) -> List:
    dict_models = {
        "LogisticRegression": LogisticRegression(max_iter=class_params.max_iter,
                                                 penalty=class_params.penalty),

        "RandomForestClassifier": RandomForestClassifier(n_estimators=class_params.n_estimators,
                                                         criterion=class_params.criterion,
                                                         max_depth=class_params.max_depth)
    }

    if class_params.type not in dict_models.keys():
        raise Exception(f"The model type: {class_params.type} is not supported.")

    return dict_models[class_params.type]


class Model:
    def __init__(self, models_params: ModelsParams,
                 class_params: ClassifierParams,
                 logger: Logger):
        self.models_params = models_params
        self.clf_params = class_params
        self.clf = None
        self.logger = logger

        self.logger.info('--- model created')

    def fit(self, X: np.array, y: np.array) -> "Model":
        self.logger.info('--- fit model begin')

        clf = get_clf(self.clf_params)
        clf.fit(X, y)

        self.clf = clf
        joblib.dump(clf, self.models_params.output_model_path)
        self.logger.info('--- fit model done')

    def predict(self, X: np.array, path: str):
        if self.clf is None:
            self.load_from_file()

        self.logger.info('--- model predict begin')
        predicts = self.clf.predict(X)
        self.logger.info('--- model predict done')

        pd.DataFrame(predicts).to_csv(path)

    def load_from_file(self):
        print(self.models_params.output_model_path)
        self.clf = joblib.load(self.models_params.output_model_path)

    def evaluate(
            self, X: np.ndarray, y: np.array, path: str
            ) -> Dict[str, float]:
        self.logger.info('--- model evaluate begin')

        if not self.clf:
            self.load_from_file()

        target = self.clf.predict(X)
        metrics_dict = {
            "f1_score": f1_score(target, y),
            "accuracy": accuracy_score(target, y),
            "roc_auc_score": roc_auc_score(target, y),
        }

        with open(path, "w") as file:
            json.dump(metrics_dict, file)

        self.logger.info('--- model evaluate done')
