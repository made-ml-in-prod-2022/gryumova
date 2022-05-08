import numpy as np
import pandas as pd
from logging import Logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.enities.feature_params import FeatureParams


class FeatureBuilder:
    def __init__(self, features: list, params: FeatureParams, logger: Logger):
        self.features = features
        self.params = params
        self.logger = logger

        self.logger.info('--- transformer build')
        self.transformer = self.build_transformer(params)
        self.logger.info('--- done')

    def build_categorical_pipeline(self, params: FeatureParams) -> Pipeline:
        self.logger.info("---- categorial pipeline build")
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy=params.missing_categorial)),
                ("ohe", OneHotEncoder()),
            ]
        )
        return categorical_pipeline

    def build_numerical_pipeline(self, params: FeatureParams) -> Pipeline:
        self.logger.info("---- numerical pipeline build")
        num_pipeline = Pipeline(
            [("impute", SimpleImputer(missing_values=np.nan, strategy=params.missing_numeric)),
             ("scaler", StandardScaler())]
        )
        return num_pipeline

    def make_features(self, transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
        return transformer.transform(df)

    def build_transformer(self, params: FeatureParams) -> Pipeline:
        transformer = Pipeline(
            [
                (
                    "column_pipeline",
                    ColumnTransformer(
                        [
                            (
                                "categorical_pipeline",
                                self.build_categorical_pipeline(params),
                                params.categorical_features,
                            ),
                            (
                                "numerical_pipeline",
                                self.build_numerical_pipeline(params),
                                params.numerical_features,
                            )
                        ])
                )
            ]
        )
        return transformer

    def extract_target(self, df: pd.DataFrame, params: FeatureParams) -> pd.Series:
        target = df[params.target_col]
        if params.use_log_trick:
            target = pd.Series(np.log(target.to_numpy()))

        return target
