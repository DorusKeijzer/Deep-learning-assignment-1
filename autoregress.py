from scipy.io import loadmat
import torch
from torch import load
import numpy as np
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler if you prefer
import click

from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel
from model_architectures.one_dim_cnn import CNN1DModel
from model_architectures.tcn import TCNModel
from model_architectures.rnn import SimpleRNNModel
from model_architectures.lstm import LSTMModel


@click.command()
@click.option('--model_path', '-m', 
              multiple=True,
              type=str,
              help='Path of state dict of the model to autoregress on'
              )
@click.option('--lag_param', '-l', 
              type=int,
              help='Lag parameter window size expected by this model'
              )

def main(model_path: str, lag_param: int, scaler_type: str):
    model = GRUModel()  
    model.load_state_dict(load(model_path))
    model.eval()  

    data = list(np.array(loadmat("data/Xtrain.mat")["Xtrain"]).flatten())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    input_data = torch.tensor(scaled_data[-lag_param:], dtype=torch.float32).unsqueeze(0)  # Shape: [1, lag_param]
    
    predictions = []  

    for _ in range(200):
        output = model(input_data)

        # reverse the scaling to get the prediction in the original scale
        output_value = output.item()  
        original_output = scaler.inverse_transform(np.array([[output_value]]))[0][0]
        predictions.append(original_output)
        
        input_data = torch.cat((input_data[:, 1:], torch.tensor([[output_value]], dtype=torch.float32)), dim=1)

    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    main()

