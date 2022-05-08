from dataclasses import dataclass


@dataclass
class ClassifierParams:
    type: str = ""
    loss: str = ""
    penalty: str = ""
    alpha: float = 0
    max_iter: int = 0
    n_estimators: int = 0
    criterion: str = ""
    max_depth: int = 0
