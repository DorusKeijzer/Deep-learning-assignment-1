from collections import defaultdict
from collections.abc import Callable
import re
import sys
from datetime import datetime

from scipy.io._fast_matrix_market import os
from torchvision.utils import math
from utils.dataloader import create_datasets_and_loaders
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import baselines, GRUs, OneDCNNs, TCNs, RNNs, LSTMs, TESTGRU #, etc.
from model_architectures.base_model import BaseModel
import click
import torch
from torch import nn
from typing import List, Tuple
    
from datetime import datetime
from typing import List, Tuple, Type, Dict, Any
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

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

def parse_lag_params(lag_params: str):
    parsed_params = []
    
    for param in lag_params.split(","):
        if "-" in param:  # A range is given
            range_parts = param.split("-")
            start = int(range_parts[0])
            end = int(range_parts[1])
            step = 1  # Default step size
            
            if ":" in range_parts[1]:  # Check if a step size is provided
                end, step = map(int, range_parts[1].split(":"))
            
            parsed_params.extend(range(start, end + 1, step))
        else:
            parsed_params.append(int(param))
    
    return parsed_params


@click.command()
@click.option('--models', '-m', 
              multiple=True,
              type=click.Choice(['LSTM', 'GRU', '1DCNN', 'TCN', 'RNN', "testGRU"], case_sensitive=False), 
              help='Model names to train (can specify multiple)')
@click.option('--lag_params', '-l',
              default="5",
              help='Lag parameters to test. Accepts comma-separated values or ranges, e.g. "5,6,8" or "5-10" or "5,6,8-10:2"')
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
@click.option('--runs', '-r',
              type=int,
              default=3,
              show_default=True,
              help="Number of times to repeat each experiment (averages out randomness).")
def main(models: List[str],
         lag_params: str, 
         datasets: List[str], 
         epochs: int,
         learning_rate: float,
         no_early_stopping: bool,
         no_show: bool,
         runs: int,) -> None:          

    lag_params = parse_lag_params(lag_params)
    show_plots = not no_show

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"Saving results to: {results_dir}")

    log_path = os.path.join(results_dir, 'run.log')
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    try:
        models_to_evaluate: List[Tuple[Callable, Dict[str, Any]]] = []  
        summary_results = []  # To store (mean_score, model_name, lag_param, dataset)


        if "1DCNN" in models:
            models_to_evaluate.extend(OneDCNNs)
        if "GRU" in models:
            models_to_evaluate.extend(GRUs)
        if "baselines" in models:
            models_to_evaluate.extend(baselines)
        if "TCN" in models:
            models_to_evaluate.extend(TCNs)
        if "testGRU" in models:
            models_to_evaluate.extend(TESTGRU)
        if "LSTM" in models:
            models_to_evaluate.extend(LSTMs)
        if "RNN" in models:
            models_to_evaluate.extend(RNNs)

        if len(models_to_evaluate) == 0:
            models_to_evaluate = ALL_MODELS

        click.echo(f"Training models:")
        for model_class, params in models_to_evaluate:
            temporary_model = create_model(1, model_class, params)
            print(f"\t{temporary_model.name}:")
            print(f"\t  {temporary_model.parameters}")
        click.echo(f"Using lag parameters: {', '.join(map(str, lag_params))}")
        click.echo(f"On datasets: {', '.join(map(str, datasets))}")
        click.echo(f"For {epochs} epochs, with {'no ' if no_early_stopping else ''}early stopping.")

        for model_class, params in models_to_evaluate:
            scores = defaultdict(lambda: defaultdict(list))  # scores[dataset][lag] = [score1, score2, score3...]


            for lag_param in lag_params:
                for dataset in datasets:
                    for run in range(runs):
                        padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = create_datasets_and_loaders(lag_param)
                        if dataset == "Non-padding":
                            test_loader, train_loader = nonpadding_test_loader, nonpadding_train_loader
                        elif dataset == "Padding":
                            test_loader, train_loader = padding_test_loader, padding_train_loader
                        else:
                            raise Exception(f"'{dataset}' is not a valid dataset. Dataset can only be 'Padding' or 'Non-padding'")
                        
                        model = create_model(lag_param, model_class, params)
                        if not model.is_baseline:
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        else:
                            optimizer = None

                        print(f"Run {run+1}/{runs}: Training {model.name} on {dataset} with lag {lag_param}")
                        score = train(model, train_loader, test_loader, learning_rate, epochs, LOSS_FUNC, optimizer, dataset, lag_param, plots_dir, weights_dir, show_plots)
                        scores[dataset][lag_param].append(score)



                            # Only collect average after the final run for that config
                        if run == runs - 1:
                            mean_score = np.mean(scores[dataset][lag_param])
                            summary_results.append((mean_score, model.name, lag_param, dataset))





        fig, ax = plt.subplots(figsize=(12, 8)) 

        for dataset_type, dataset_scores in scores.items():
            avg_scores = [np.mean(dataset_scores[lag]) for lag in lag_params]
            std_scores = [np.std(dataset_scores[lag]) for lag in lag_params]
            ax.errorbar(lag_params, avg_scores, yerr=std_scores, capsize=5, linewidth=2, marker='o', label=dataset_type)

        ax.set_xlabel("Lag Parameter", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=12)
        ax.set_title(f"Validation score vs Lag for {model.name}", fontsize=14, pad=20)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True)
        ax.legend(fontsize=10)

        summary_text = (
            r"$\bf{Model\ name}$" + f": {model.name}\n"
            r"$\bf{Parameters}$" + f": {model.model_parameters}\n"
            r"$\bf{Evaluated\ Datasets}$" + f": {', '.join(scores.keys())}\n"
            r"$\bf{Lag\ Params}$" + f": {lag_params}\n"
        )

        fig.text(
            0.1, 0.01,
            summary_text,
            ha='left',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray')
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25)  

        plot_path = os.path.join(plots_dir, f"{model.name}_lag_performance.png")
        if show_plots:
            plt.show()

        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved lag performance plot to {plot_path}")
        plt.close(fig)



                # Print all configurations sorted by average validation error
        print("\n=== Sorted Results by Average Validation Score ===")
        summary_results.sort()  # Sort by mean_score ascending
        for mean_score, model_name, lag_param, dataset in summary_results:
            print(f"{model_name + ' ' + model.model_parameters:<25} | Dataset: {dataset:<12} | Lag: {lag_param:<3} | Avg Score: {mean_score:.4f}")



    finally:
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
         no_early_stopping: bool = False,) -> float:


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
        return np.min(avg_val_losses)

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
