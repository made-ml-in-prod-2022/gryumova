from dataclasses import dataclass


@dataclass()
class LoggerParams:
    path: str = 'online_inference.log'
    format: str = '%(asctime)s %(levelname)s %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    level: int = 20
    mode: str = ''
    stdout: bool = True
