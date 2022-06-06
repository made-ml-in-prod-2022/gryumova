from typing import List
import pandas as pd
import uvicorn
import click

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from core import ServiceCore
from entities.med_request import MedicalRequest
from entities.med_response import MedicalResponse
from entities.app_params import read_app_params

from entities.app_params import read_app_params

from utils import create_logger


app = FastAPI()
core = ServiceCore()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get('/')
def root():
    return 'This is entry point of our predictor'


@app.get('/touch')
def touch() -> bool:
    return core.is_init and (core.model is not None)


@app.post('/predict/', response_model=List[MedicalResponse])
def predict(request: MedicalRequest) -> List[MedicalResponse]:
    try:
        core.logger.info('processing request...')
        data_js = request.data
        data_pd = pd.DataFrame(data_js)
        print(data_pd.shape)
        preds = core.predict(data_pd)
        core.logger.info('request precessed successfully')

        return [MedicalResponse(result=int(r)) for r in preds]
    except Exception as ex:
        core.logger.error(f'Critical model error: {str(ex)}')
        raise HTTPException(status_code=500,
                            detail='Critical error during making predictions')


@click.command(name='main')
@click.argument('config-path', default='../configs/app_configs.yaml')
def main(config_path: str):
    app_params = read_app_params(config_path)
    MedicalRequest.real_features = app_params.features

    logger = create_logger('inference', app_params.logging)

    core.init(app_params, logger)

    core.logger.info('starting service...')
    uvicorn.run(app, host=app_params.host, port=app_params.port)
    core.logger.info('service successfully finished')


if __name__ == '__main__':
    main()
