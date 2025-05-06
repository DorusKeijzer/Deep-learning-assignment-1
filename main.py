from collections import defaultdict
import json
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

def parse_lag_params(lag_params: str) -> list[int]:
    """
    Parse a string like "5,6,8-12:2,20" into a sorted list of ints:
      - single values: "5"
      - ranges:        "8-12"
      - ranges+step:   "5-20:5"
    """
    parsed: set[int] = set()

    for token in lag_params.split(","):
        token = token.strip()
        if "-" not in token:
            # single value
            parsed.add(int(token))
        else:
            # range (maybe with step)
            # split into "start" and "rest"
            start_str, rest = token.split("-", 1)
            start = int(start_str)
            if ":" in rest:
                # form "end:step"
                end_str, step_str = rest.split(":", 1)
                end = int(end_str)
                step = int(step_str)
            else:
                end = int(rest)
                step = 1

            if step <= 0:
                raise ValueError(f"Step must be positive in '{token}'")
            # build inclusive range
            for v in range(start, end + 1, step):
                parsed.add(v)

    return sorted(parsed)

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
        raw_results = []

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



            scores_by_model_and_dataset = {}

            for lag_param in lag_params:
                for dataset in datasets:
                    for run in range(runs):
                        (loaders, scaler) = create_datasets_and_loaders(lag_param)
                        padding_train_loader, padding_test_loader, nonpadding_train_loader, nonpadding_test_loader = loaders
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

                        model_key = f"{model.name}_{model.model_parameters}_{dataset}"
                        if model_key not in scores_by_model_and_dataset:
                            scores_by_model_and_dataset[model_key] = {}

                        print(f"Run {run+1}/{runs}: Training {model.name} on {dataset} with lag {lag_param}")
                        score = train(
                            model, train_loader, test_loader,
                            learning_rate, epochs, LOSS_FUNC,
                            optimizer, dataset, lag_param,
                            plots_dir, weights_dir, show_plots, scaler, params
                        )

                        test_run_result = {
                            "model_name": model.name,
                            "model_parameters": model.model_parameters,
                            "dataset": dataset,
                            "lag_param": lag_param,
                            "run": run + 1,
                            "score": float(score)
                        }
                        
                        raw_results.append(test_run_result)



                        scores[dataset][lag_param].append(score)

                        scores_by_model_and_dataset[model_key].setdefault(lag_param, []).append(score)

                        if run == runs - 1:
                            mean_score = np.mean(scores[dataset][lag_param])
                            summary_results.append((mean_score, model.name, lag_param, dataset))
                    fig, ax = plt.subplots(figsize=(12, 8)) 

            for model_key, lag_scores_dict in scores_by_model_and_dataset.items():
                lags_sorted = sorted(lag_scores_dict.keys())
                avg_scores = [np.mean(lag_scores_dict[lag]) for lag in lags_sorted]
                std_scores = [np.std(lag_scores_dict[lag]) for lag in lags_sorted]

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.errorbar(lags_sorted, avg_scores, yerr=std_scores, capsize=5, linewidth=2, marker='o')
                ax.set_xlabel("Lag Parameter", fontsize=12)
                ax.set_ylabel("Average Score", fontsize=12)
                ax.set_title(f"Lag Performance for {model_key}", fontsize=14, pad=20)
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=10)

                # Extract model name and dataset for summary
                parts = model_key.split("_")
                model_name = parts[0]
                dataset_type = parts[-1]
                param_str = "_".join(parts[1:-1])

                summary_text = (
                    r"$\bf{Model\ name}$" + f": {model_name}\n"
                    r"$\bf{Parameters}$" + f": {param_str}\n"
                    r"$\bf{Dataset}$" + f": {dataset_type}\n"
                    r"$\bf{Lag\ Params}$" + f": {lags_sorted}"
                )
                fig.text(
                    0.1, 0.01,
                    summary_text,
                    ha='left',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray')
                )

                # Save to unique file
                safe_model_key = model_key.replace(" ", "_").replace(",", "").replace("=", "-")
                filename = f"lag_performance_{safe_model_key}.png"
                plot_path = os.path.join(plots_dir, filename)
                plt.tight_layout()
                fig.subplots_adjust(bottom=0.25)
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"Saved per-model lag performance plot to {plot_path}")
                plt.close(fig)
             
                # Print all configurations sorted by average validation error
        print("\n=== Sorted Results by Average Validation Score ===")
        summary_results.sort()  # Sort by mean_score ascending
        for mean_score, model_name, lag_param, dataset in summary_results:
            print(f"{model_name + ' ' + model.model_parameters:<25} | Dataset: {dataset:<12} | Lag: {lag_param:<3} | Avg Score: {mean_score:.4f}")

                
        json_path = os.path.join(results_dir, "raw_test_results.json")
        with open(json_path, "w") as f:
            json.dump(raw_results, f, indent=2)

        print(f"Saved raw test results to {json_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def plot_raw_losses(scaled_train_mse: list[float],
                    raw_val_mse: list[float],
                    model: BaseModel,
                    dataset: str,
                    run_name: str,
                    lag: int,
                    plots_dir: str,
                    scaler: object,
                    show_plots: bool):
    sigma = float(scaler.scale_[0])

    raw_train_rmse = [math.sqrt(m) * sigma for m in scaled_train_mse]
    raw_val_rmse = [math.sqrt(m) for m in raw_val_mse]

    epochs = list(range(1, len(raw_train_rmse) + 1))
    
    # Rest of the plotting code remains the same...
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(epochs, raw_train_rmse, label="Train RMSE", color="blue")
    ax.plot(epochs, raw_val_rmse,   label="Val   RMSE", color="orange")

    # Highlight best epochs
    best_train_ep = int(np.argmin(raw_train_rmse) + 1)
    best_val_ep   = int(np.argmin(raw_val_rmse)   + 1)
    ax.plot(best_train_ep, raw_train_rmse[best_train_ep-1], 'o', markersize=8, label="Best Train", color="blue")
    ax.plot(best_val_ep,   raw_val_rmse[best_val_ep-1],     'o', markersize=8, label="Best Val", color="orange")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("RMSE (original units)", fontsize=12)
    ax.set_title(f"{model.name} on {dataset} (lag={lag})", fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True)

    # Text box summary
    summary = (
        f"Model: {model.name}\n"
        f"Params: {model.model_parameters}\n"
        f"Lag: {lag}\n"
        f"Best Train RMSE: {raw_train_rmse[best_train_ep-1]:.3f} (ep {best_train_ep})\n"
        f"Best Val   RMSE: {raw_val_rmse[best_val_ep-1]:.3f} (ep {best_val_ep})"
    )
    fig.text(0.1, 0.02, summary, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    out_path = os.path.join(plots_dir, f"{run_name}_raw_rmse.png")
    if show_plots:
        plt.show()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved raw‐unit RMSE plot to {out_path}")

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
          optimizer: torch.optim.Optimizer,
          dataset: str,
          lag: int,
          plots_dir: str,
          weights_dir: str,
          show_plots: bool,
          scaler: object,
          model_params: Dict[str, Any],
          no_early_stopping: bool = False) -> float:

    run_name = generate_run_name(model)

    # Lists to collect scaled MSE per epoch
    scaled_train_mse = []
    scaled_val_mse   = []

    best_scaled_mse = float('inf')
    patience = 5
    min_delta = 0.0001
    epochs_no_improve = 0
    best_state = None


# Inside the train function
    scaled_val_mse = []
    raw_val_mse = []

    for epoch in range(1, epochs + 1):
        scaled_mse_train = train_loop(model, train_loader, learning_rate, loss_fn, optimizer)
        scaled_train_mse.append(scaled_mse_train) 
        # Validation
        scaled_mse_val, raw_mse_val = test_loop(model, test_loader, loss_fn, scaler)
        scaled_val_mse.append(scaled_mse_val)
        raw_val_mse.append(raw_mse_val)

        # Early stopping based on scaled MSE
        if scaled_mse_val < best_scaled_mse - min_delta:
            best_scaled_mse = scaled_mse_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if not no_early_stopping and epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    # Report best scaled losses
    best_train = min(scaled_train_mse)
    best_val   = min(scaled_val_mse)
    print(f"\nBest scaled train MSE: {best_train:.6f} (epoch {np.argmin(scaled_train_mse)+1})")
    print(f"Best scaled  val MSE: {best_val:.6f} (epoch {np.argmin(scaled_val_mse)+1})")

    # Plot RMSE in original units
    plot_raw_losses(
        scaled_train_mse,
        raw_val_mse,  
        model, 
        dataset,
        run_name, 
        lag,
        plots_dir,
        scaler,
        show_plots
  )
    # Save best model weights
    save_path = os.path.join(weights_dir, run_name + ".pt")
    model_package = {
        "model_class": model.__class__.__name__,
        "model_parameters": model_params,  
        "state_dict": best_state
    }
    torch.save(model_package, save_path)

    # Return raw MSE for sorting/summary
    _, raw_mse = test_loop(model, test_loader, loss_fn, scaler)
    return raw_mse

def train_loop(model: BaseModel,
               train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
               learning_rate: float,
               loss_fn: Type[nn.MSELoss],
               optimizer: torch.optim.Optimizer) -> float:

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        # x: [batch, lag]  → [batch, 1, lag] if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # y: [batch] → [batch,1]
        if y.dim() == 1:
            y = y.unsqueeze(1)

        pred = model(x)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        if batch_idx % 5 == 0:
            avg = total_loss / total_samples
            print(f"loss: {avg:7f}  [{total_samples:5d}/{len(train_loader.dataset):5d}]")

    epoch_loss = total_loss / total_samples
    print(f"Epoch complete - average loss: {epoch_loss:.6f}")
    return epoch_loss


def test_loop(model: BaseModel,
              test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
              loss_fn: Type[nn.MSELoss],
              scaler: object = None) -> Tuple[float, float]:
    model.eval()
    scaled_mse_accum = 0.0
    all_preds = []
    all_targs = []

    with torch.no_grad():
        for x, y in test_loader:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if y.dim() == 1:
                y = y.unsqueeze(1)

            pred = model(x)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)

            loss = loss_fn(pred, y)
            scaled_mse_accum += loss.item()

            if scaler is not None:
                p = scaler.inverse_transform(pred.cpu().numpy())
                t = scaler.inverse_transform(y.cpu().numpy())
                all_preds.append(p)
                all_targs.append(t)

    num_batches = len(test_loader)
    scaled_mse = scaled_mse_accum / num_batches
    raw_mse = scaled_mse  # Default if no scaler

    if scaler is not None and all_preds:
        all_preds = np.vstack(all_preds)
        all_targs = np.vstack(all_targs)
        raw_mse = np.mean((all_preds - all_targs) ** 2)

    return scaled_mse, raw_mse
if __name__ == '__main__':
    main()
