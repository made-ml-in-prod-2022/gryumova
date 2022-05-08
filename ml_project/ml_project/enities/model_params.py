from dataclasses import dataclass


@dataclass()
class ModelsParams:
    input_data_path: str = "heart.csv"
    output_model_path: str = "model.pkl"
    metric_path: str = "metrics.json"
    params_path: str = "params.json"
    save_path: str = "prediction.csv"
