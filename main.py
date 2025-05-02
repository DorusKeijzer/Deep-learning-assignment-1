from collections.abc import Callable
import re
import sys
from datetime import datetime

from scipy.io._fast_matrix_market import os
from model_architectures.gru import GRUModel
from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from torch.nn.modules import MSELoss
from torch.optim import optimizer
from torchvision.utils import math
from model_architectures import baseline
from utils.dataloader import create_datasets_and_loaders
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import baselines, GRUs, OneDCNNs # LSTMs, etc.
from model_architectures.base_model import BaseModel
import click
import torch
from torch import nn
from typing import List, Tuple
    
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import TypeVar
from typing import List, Tuple, Type, Dict, Any
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


dont_show_plots = False


class Tee:
    """Duplicate output to both console and log file"""
    def __init__(self, *files):
        self.files = files
        
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Immediate flush for real-time logging
            
    def flush(self):
        for file in self.files:
            if hasattr(file, 'flush'):
                file.flush()



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
              type=click.Choice(['LSTM', 'GRU', '1DCNN'], case_sensitive=False),
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
@click.option('--no_show', '-n',
              is_flag=True,
              default=False,
              show_default=True,
              help="If set to True, does not show plots during runs"
              )

def main(models: List[str],
         lag_params: List[int], 
         datasets: List[str], 
         epochs: int,
         learning_rate: float,
         no_early_stopping: bool,
         no_show: bool,) -> None:          

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
        no_show: if True, do not show plots between training runs (saves manual clicking). 

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

    show_plots = not no_show

    # creating save locations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)


    os.makedirs(results_dir, exist_ok=True)

    plots_dir = os.path.join(results_dir, "plots")
    weights_dir = os.path.join(results_dir, "weights")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"Saving results to: {results_dir}")

    print(f"Saving results to: {results_dir}")

    # Set up logging to file
    log_path = os.path.join(results_dir, 'run.log')
    log_file = open(log_path, 'w')
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect all output to logging file as well
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    try:


        models_to_evaluate: List[Tuple[Callable, Dict[str, Any]]] = []  

        if "1DCNN" in models:
            models_to_evaluate.extend(OneDCNNs)
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
            print(f"\t{temporary_model.name}:")
            print(f"\t  {temporary_model.parameters}")
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
                    print(f"Training and evaluating model: {model.name} {model.model_parameters} on dataset: {dataset}")

                    train(model,train_loader, test_loader, learning_rate, epochs, LOSS_FUNC, optimizer, dataset, lag_param, plots_dir, weights_dir, show_plots)

    finally:
        # Restore original output streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()        



def plot_losses(avg_train_losses: List[float], 
                avg_val_losses: List[float], 
                model: BaseModel,
                dataset: str,
                run_name: str,
                lag: int,
                plots_dir: str,
                show_plots: bool):
    
    # Create larger figure with adjusted aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))  # Width: 12", Height: 8"
    
    # Plot loss curves with thicker lines
    ax.plot(range(len(avg_train_losses)), avg_train_losses, 
            label="Train", linewidth=2)
    ax.plot(range(len(avg_val_losses)), avg_val_losses, 
            label="Validation", linewidth=2)

    # Highlight minimum loss points with larger markers
    best_train_epoch = np.argmin(avg_train_losses)
    best_val_epoch = np.argmin(avg_val_losses)

    ax.plot(best_train_epoch, avg_train_losses[best_train_epoch], 
            'o', color='blue', markersize=8, label='Best Train')
    ax.plot(best_val_epoch, avg_val_losses[best_val_epoch], 
            'o', color='orange', markersize=8, label='Best Val')

    # Increase font sizes
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(f"Training curve of {model.name} on {dataset.lower()} dataset (lag={lag})", 
                fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Text box formatting
    fig.text(
        0.1, 0.01, 
        r"$\bf{Model\ name}$" + f": {model.name}\n"
        r"$\bf{Lag}$" + f": {lag}\n"
        r"$\bf{Parameters}$" + f": {model.model_parameters}\n"
        r"$\bf{Run}$" + f": {run_name}\n"
        r"$\bf{Min\ Train\ Loss}$" + f": {avg_train_losses[best_train_epoch]:.2f} (epoch {best_train_epoch})\n"
        r"$\bf{Min\ Val\ Loss}$" + f": {avg_val_losses[best_val_epoch]:.2f} (epoch {best_val_epoch})\n",
        ha='left', 
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray')
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # More space for text box

    if show_plots:
        plt.show()
        
    save_location = os.path.join(plots_dir, f"{run_name}.png")
    plt.savefig(save_location, bbox_inches='tight', dpi=300)  # High-res save
    plt.close(fig)  # Important: prevent memory leaks



def generate_run_name(model: BaseModel):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_{model.name}_{model.model_parameters}"
    return re.sub(r":", "_", re.sub(r"\s+", "_", name))

def train(model: BaseModel,
         train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
         test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
         learning_rate: float, 
         epochs: int,
         loss_fn: Type[nn.MSELoss],
         optimizer: torch.optim.Adam,
         dataset: str,
           lag: int,
          plots_dir: str,
          weights_dir: str,
          show_plots: bool,
         no_early_stopping: bool = False,) -> None:


    run_name = generate_run_name(model)

    if not model.is_baseline:  # non baseline models need training
        # Early stopping parameters
        patience = 5
        min_delta = 0.001
        best_loss = float('inf')
        epochs_without_improvement = 0

        # to store average loss per epoch
        avg_train_losses = []
        avg_val_losses = []
        lowest_val_loss = 10e32
        lowest_val_loss_model = None
        
        for epoch in range(epochs): 
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-------------------------------")
            
            # Training phase
            avg_train_loss = train_loop(model, train_loader, learning_rate, loss_fn, optimizer)
            avg_train_losses.append(avg_train_loss)
            
            # Validation phase
            current_loss = test_loop(model, test_loader, loss_fn, return_loss=True)
            
            avg_val_losses.append(current_loss)

            if current_loss < lowest_val_loss:
                lowest_val_loss_model = model.state_dict()


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

        print(f"lowest validation loss: {np.min(avg_val_losses)} in epoch {np.argmin(avg_val_losses)}")
        print(f"lowest training loss: {np.min(avg_train_losses)} in epoch {np.argmin(avg_train_losses)}")

        plot_losses(avg_train_losses, avg_val_losses, model, dataset, run_name, lag, plots_dir, show_plots)
        save_loacation = os.path.join(weights_dir, run_name)
        torch.save(lowest_val_loss_model, save_loacation)

    else:  # baseline models only need to be evaluated
        test_loop(model, test_loader, loss_fn)

def train_loop(model: BaseModel,
              train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
              learning_rate: float, 
              loss_fn: Type[nn.MSELoss],
              optimizer: torch.optim.Adam) -> float:
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
    return avg_epoch_loss



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
