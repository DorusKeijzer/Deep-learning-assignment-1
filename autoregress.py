import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from torch import nn
import click
from typing import Dict, Type

# Model imports
from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel
from model_architectures.one_dim_cnn import CNN1DModel
from model_architectures.tcn import TCNModel
from model_architectures.rnn import SimpleRNNModel
from model_architectures.lstm import LSTMModel


def load_model_from_file(filepath: str, model_classes: Dict[str, Type[nn.Module]]) -> nn.Module:
    package = torch.load(filepath, map_location="cpu")

    class_name = package["model_class"]
    params = package["model_parameters"]
    state_dict = package["state_dict"]

    if class_name not in model_classes:
        raise ValueError(f"Unknown model class: {class_name}")

    model_class = model_classes[class_name]
    model = model_class(**params)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@click.command()
@click.option('--model_name', '-n',
              type=click.Choice(['gru', 'lstm', 'cnn', 'rnn', 'tcn']),
              required=True,
              help='Name of the model architecture to use')
@click.option('--model_path', '-m',
              type=click.Path(exists=True),
              required=True,
              help='Path to the state dict of the model')
@click.option('--lag_param', '-l',
              type=int,
              required=True,
              help='Lag parameter window size expected by this model')
def main(model_name: str, model_path: str, lag_param: int):
    # Available models
    model_class_map = {
        'GRUModel': GRUModel,
        'LSTMModel': LSTMModel,
        'CNN1DModel': CNN1DModel,
        'SimpleRNNModel': SimpleRNNModel,
        'TCNModel': TCNModel,
        'MeanBaseline': MeanBaseline,
        'MostRecentBaseline': MostRecentBaseline,
    }



    # Load the model from the file (includes class, params, and weights)
    model = load_model_from_file(model_path, model_class_map)

    # Load and preprocess data
    data = list(np.array(loadmat("data/Xtrain.mat")["Xtrain"]).flatten())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()
    input_data = torch.tensor(scaled_data[-lag_param:], dtype=torch.float32).unsqueeze(0)

    predictions = []
    original_data_for_plotting = scaled_data.tolist()

    for _ in range(200):
        output = model(input_data)
        output_value = output.item()
        predictions.append(output_value)
        input_data = torch.cat((input_data[:, 1:], torch.tensor([[output_value]], dtype=torch.float32)), dim=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(original_data_for_plotting, color='blue', label="Original Data")
    plt.plot(range(len(original_data_for_plotting), len(original_data_for_plotting) + len(predictions)), predictions, color='red', label="Autoregressive Prediction")

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f"Original Data and Autoregressive Predictions for {model_name.upper()}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
