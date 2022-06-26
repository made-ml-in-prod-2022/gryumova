import os
import pickle
import json
import pandas as pd
import click
from sklearn.metrics import accuracy_score


@click.command('model_validate')
@click.option('--input-path')
@click.option('--output-path')
def model_validate(input_path: str, output_path: str):
    data_valid_path = os.path.join(input_path, 'data_valid.csv')
    
    model_path = os.path.join(output_path, 'model.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_valid = pd.read_csv(data_valid_path)
    X_valid = data_valid.drop('target', axis=1)
    y_valid = data_valid['target'].values

    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    with open(os.path.join(output_path, 'valid_stats.json'), 'w') as f:
        json.dump({'accuracy': accuracy}, f)


if __name__ == '__main__':
    model_validate()
