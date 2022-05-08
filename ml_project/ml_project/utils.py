from logging import Logger
import logging
import os

from ml_project.enities.logger_params import LoggerParams


def make_logger(name: str, params: LoggerParams) -> Logger:
    folder = os.path.dirname(params.path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_log = logging.FileHandler(params.path)
    console_out = logging.StreamHandler()

    logging.basicConfig(handlers=(file_log, console_out),
                        format=params.format,
                        datefmt=params.date_format,
                        level=logging.INFO)

    logger = logging.getLogger(name)

    return logger


def get_logger(name: str, params: LoggerParams) -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return make_logger(name, params)

    return logger
