from dataclasses import dataclass


@dataclass()
class LoggerParams:
    name: str = "ml_project_hw1"
    path: str = 'ml_project.log'
    format: str = '%(asctime)s %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    stdout: bool = True
