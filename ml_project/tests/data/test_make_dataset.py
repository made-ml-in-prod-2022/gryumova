from ml_project.data.make_dataset import read_data, split_tain_val_data
from ml_project.enities.split_params import SplittingParams


def test_load_dataset(input_data_path: str, target_col: str):
    data = read_data(input_data_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(input_data_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    data = read_data(input_data_path)
    train, val = split_tain_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10
