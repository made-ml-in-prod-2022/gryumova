import os
import pickle
import pandas as pd
import click


@click.command('model_predict')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--model-path')
def model_predict(input_path: str, output_path: str, model_path: str):
    data_path = os.path.join(input_path, 'data.csv')
    
    os.makedirs(output_path, exist_ok=True)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_pred = pd.read_csv(data_path)
    y_pred = model.predict(data_pred)
    y_pred = pd.DataFrame({'pred': y_pred})
    y_pred.to_csv(os.path.join(output_path, 'predictions.csv'))


if __name__ == '__main__':
    model_predict()
