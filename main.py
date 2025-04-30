from collections.abc import Callable
from model_architectures import baseline
from utils.dataloader import create_datasets_and_loaders
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import baselines, GRUs # LSTMs, etc.
from model_architectures.base_model import BaseModel
import click
import torch
from typing import List, Tuple
    
from sklearn.model_selection import train_test_split



# argument parsing
@click.command()
@click.option('--models', '-m', 
              multiple=True,
              type=click.Choice(['LSTM', 'GRU', 'TCN', 'Transformer'], case_sensitive=False),
              help='Model names to train (can specify multiple)')
@click.option('--lag_params', '-l',
              multiple=True,
              default=[5],
              type=int,
              help='Lag parameters to test (can specify multiple)')
@click.option('--datasets', '-d',
              multiple=True, 
              type=click.Choice(['Padding', 'Non-padding'], case_sensitive=False),
              default = ['Padding', 'Non-padding'],
              help="Use dataset with padding or without (or both)")
@click.option('--epochs', '-e',
              type=int,
              default=100,
              show_default=True,
              help="Number of epochs to train"
              )
@click.option('--learning_rate', '-lr',
              type=float,
              default=1e-3,
              show_default=True,
              help="Learning rate."
              )
@click.option('--no_early_stopping', '-s',
              is_flag=True,
              default=False,
              show_default=True,
              help="If set to True, finishes all specified epochs, else stops early"
              )


def main(models: List[str],
         lag_params: List[int], 
         datasets: List[str], 
         epochs: int,
         learning_rate: float,
         no_early_stopping: bool, 
         ):
    """
    Train and evaluate models with different lag parameters on specified datasets.
    
    Args:
        models: List of models to train. Choices are 'LSTM', 'GRU', 'TCN', or 'Transformer'.
                If not specified, all models will be trained.
        lag_params: List of lag parameters to test. Default is [5] if not specified.
        datasets: List of dataset types to use. Choices are 'Padding' or 'Non-padding'.
                  Default is both ['Padding', 'Non-padding'].
        epochs: Number of training epochs. Default is 100.
        learning_rate: Learning rate for training. Default is 1e-3.
        no_early_stopping: If True, runs all epochs without early stopping.
                          If False (default), may stop early based on validation performance.

    Notes:
        - Multiple models, lag parameters, and datasets can be specified by repeating the flags
        - Model names and dataset types are case-insensitive

    Examples:
        # Train LSTM and GRU with lags 5 and 10
        python main.py -m LSTM -m GRU -l 5 -l 10
        
        # Train Transformer with lag 5 on Padding dataset
        python main.py --models Transformer --lag_params 5 --dataset Padding
        
        # Train on both dataset types for 50 epochs without early stopping
        python main.py -d Padding -d Non-padding --epochs 50 --no-early-stopping
        
        # Train all models with default parameters
        python main.py
    """

    
    models_to_evalute = []
    if "GRU" in models:
        models_to_evalute.extend(GRUs)
    if "baselines" in models:
        models_to_evalute.extend(baselines)
    # etc. TODO: add when available

    if len(models_to_evalute) == 0:
        models_to_evalute = ALL_MODELS
    
    click.echo(f"Training models:")
    for model in models_to_evalute:
        print(f"\t{model.name}")
    click.echo(f"Using lag parameters: {', '.join(map(str, lag_params))}")
    click.echo(f"On datasets: {', '.join(map(str, datasets))}")
    click.echo(f"For {epochs} epochs, with {"no " if no_early_stopping else ""}early stopping.")

    for model in models_to_evalute:
        for lag_param in lag_params:
            for dataset in datasets:
                padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = create_datasets_and_loaders(lag_param)
                if dataset == "Non-padding":
                    train(model, nonpadding_train_loader, nonpadding_test_loader, epochs, not no_early_stopping, learning_rate)
                if dataset == "Padding":
                    train(model, padding_train_loader, padding_test_loader, epochs, not no_early_stopping, learning_rate)

                    



def train(model: BaseModel, 
          train_loader: torch.utils.data.DataLoader[any], 
          test_loader: torch.utils.data.DataLoader[any],
          epochs: int,
          early_stopping: bool,
          learning_rate: float): 
    """
    Main training loop. 
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        


if __name__ == '__main__':
    main()




