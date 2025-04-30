from collections.abc import Callable

from torch.nn.modules import MSELoss
from torch.optim import optimizer
from model_architectures import baseline
from utils.dataloader import create_datasets_and_loaders
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import baselines, GRUs # LSTMs, etc.
from model_architectures.base_model import BaseModel
import click
import torch
from torch import nn
from typing import List, Tuple
    
from sklearn.model_selection import train_test_split
from typing import TypeVar
from typing import List, Tuple, Type
import torch
from torch.utils.data import DataLoader
from torch import nn

# loss function is MSE everywhere
LOSS_FUNC = nn.MSELoss()



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
         no_early_stopping: bool) -> None:          

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
    models_to_evaluate: List[BaseModel] = []  

    if "GRU" in models:
        models_to_evaluate.extend(GRUs)
    if "baselines" in models:
        models_to_evaluate.extend(baselines)
    # etc. TODO: add when available

    if len(models_to_evaluate) == 0:
        models_to_evaluate = ALL_MODELS
    
    click.echo(f"Training models:")
    for model in models_to_evaluate:
        print(f"\t{model.name}")
    click.echo(f"Using lag parameters: {', '.join(map(str, lag_params))}")
    click.echo(f"On datasets: {', '.join(map(str, datasets))}")
    click.echo(f"For {epochs} epochs, with {'no ' if no_early_stopping else ''}early stopping.")

    for model in models_to_evaluate:
        if not model.is_baseline:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else: 
            optimizer = None

        for lag_param in lag_params:
            for dataset in datasets:
                padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = create_datasets_and_loaders(lag_param)
                if dataset == "Non-padding":
                    test_loader, train_loader = nonpadding_test_loader, nonpadding_train_loader
                if dataset == "Padding":
                    test_loader, train_loader = padding_test_loader,    padding_train_loader
                else:
                    raise Exception("Dataset can only be 'Padding' or 'Non-padding'")
                train(model,train_loader, test_loader, learning_rate, epochs, LOSS_FUNC, optimizer)
                
def train(model: BaseModel,
          train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
          test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
          learning_rate: float, 
          epochs: int,
          loss_fn: Type[nn.MSELoss],
          optimizer: torch.optim.Adam) -> None:

    print(f"Training and evaluating model: {model.name}")
    if not model.is_baseline: # non baseline models need training
        for epoch in range(epochs): 
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loop(model, train_loader, learning_rate, loss_fn, optimizer)
            test_loop(model, test_loader, loss_fn)

    else: # baseline models only need to be evaluated
        test_loop(model, test_loader, loss_fn)


def train_loop(model: BaseModel,
       train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
       learning_rate: float, 
       loss_fn: Type[nn.MSELoss],
       optimizer: torch.optim.Adam) -> None:
    """
    Main training loop. 
    """

    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    size = len(train_loader.dataset)

    for batch, (x, y) in enumerate(train_loader): 
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * 32 + len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(model: BaseModel,
              test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
              loss_fn: Type[nn.MSELoss]) -> None:
    test_loss: float = 0
    correct: float = 0


    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


if __name__ == '__main__':
    main()




