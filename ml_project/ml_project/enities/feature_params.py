from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
    missing_categorial: str = "most_frequent"
    missing_numeric: str = "mean"
    use_log_trick: bool = False
