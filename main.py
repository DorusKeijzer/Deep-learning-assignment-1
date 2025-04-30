from collections.abc import Callable
from model_architectures.gru import GRUModel
from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from torch.nn.modules import MSELoss
from torch.optim import optimizer
from torchvision.utils import math
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
from typing import List, Tuple, Type, Dict, Any
import torch
from torch.utils.data import DataLoader
from torch import nn

# loss function is MSE everywhere
LOSS_FUNC = nn.MSELoss()

def create_model(lag_param: int, model_class, params: Dict[str, Any]):
    """creates a model suitable to the current input sise of the specified class using the specified parameter"""
    params["input_size"] = lag_param
    
    # Instantiate the model using parameters in the dictionary
    return model_class(**params)




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
    models_to_evaluate: List[Tuple[BaseModel, Dict[str, Any]]] = []  

    if "GRU" in models:
        models_to_evaluate.extend(GRUs)
    if "baselines" in models:
        models_to_evaluate.extend(baselines)
    # etc. TODO: add when available

    if len(models_to_evaluate) == 0:
        models_to_evaluate = ALL_MODELS

    click.echo(f"Training models:")
    for model_class, params in models_to_evaluate:
        # create temporary model to print model
        temporary_model = create_model(1,model_class, params)
        print(f"\t{temporary_model.name}")
    click.echo(f"Using lag parameters: {', '.join(map(str, lag_params))}")
    click.echo(f"On datasets: {', '.join(map(str, datasets))}")
    click.echo(f"For {epochs} epochs, with {'no ' if no_early_stopping else ''}early stopping.")

    for model_class, params in models_to_evaluate:

        for lag_param in lag_params:
            for dataset in datasets:
                padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = create_datasets_and_loaders(lag_param)
                if dataset == "Non-padding":
                    test_loader, train_loader = nonpadding_test_loader, nonpadding_train_loader
                elif dataset == "Padding":
                    test_loader, train_loader = padding_test_loader,    padding_train_loader
                else:
                    raise Exception(f"'{dataset}' is not a valid dataset. Dataset can only be 'Padding' or 'Non-padding'")
                model = create_model(lag_param, model_class, params)
                if not model.is_baseline:
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                else: 
                    optimizer = None
                print(f"Training and evaluating model: {model.name} on dataset: {dataset}")

                train(model,train_loader, test_loader, learning_rate, epochs, LOSS_FUNC, optimizer)

def train(model: BaseModel,
         train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
         test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
         learning_rate: float, 
         epochs: int,
         loss_fn: Type[nn.MSELoss],
         optimizer: torch.optim.Adam,
         no_early_stopping: bool = False) -> None:

    if not model.is_baseline:  # non baseline models need training
        # Early stopping parameters
        patience = 5
        min_delta = 0.001
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs): 
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-------------------------------")
            
            # Training phase
            train_loop(model, train_loader, learning_rate, loss_fn, optimizer)
            
            # Validation phase
            current_loss = test_loop(model, test_loader, loss_fn, return_loss=True)
            
            # Early stopping logic
            if not no_early_stopping:
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    epochs_without_improvement = 0
                    # Optional: save best model
                    # torch.save(model.state_dict(), 'best_model.pth')
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        print(f"Validation loss didn't improve for {patience} epochs.")
                        break
    else:  # baseline models only need to be evaluated
        test_loop(model, test_loader, loss_fn)
def train_loop(model: BaseModel,
              train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
              learning_rate: float, 
              loss_fn: Type[nn.MSELoss],
              optimizer: torch.optim.Adam) -> None:
    model.train()  # Set model to training mode
    total_loss = 0.0
    total_samples = 0
    
    for batch, (x, y) in enumerate(train_loader):
        # Ensure input has correct shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Ensure target has correct shape
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
            
        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track progress
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        
        if batch % 5 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            current = (batch + 1) * x.size(0)
            size = len(train_loader.dataset)
            print(f"loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")
    
    # Print epoch summary
    avg_epoch_loss = total_loss / total_samples
    print(f"Epoch complete - average loss: {avg_epoch_loss:.6f}")



def test_loop(model: BaseModel,
              test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
              loss_fn: Type[nn.MSELoss],
              return_loss: bool = False) -> float:
    test_loss: float = 0.0

    model.eval()
    num_batches = len(test_loader)

    with torch.no_grad():
        for x, y in test_loader:
            # Ensure input has correct shape
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            pred = model(x)
            
            # Ensure target has correct shape
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
                
            loss = loss_fn(pred, y)
            test_loss += loss.item()

    test_loss /= num_batches
    rmse = math.sqrt(test_loss)
    print(f"Root MSE: {rmse:.6f}")
    
    if return_loss:
        return test_loss  
    return rmse


if __name__ == '__main__':
    main()
