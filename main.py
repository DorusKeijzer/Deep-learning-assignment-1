from collections.abc import Callable
from model_architectures import baseline
from utils.dataloader import PaddingDataset, NonPaddingDataset
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import baselines, GRUs # LSTMs, etc.
from model_architectures.base_model import BaseModel
import click
from typing import List, Tuple

# argument parsing
@click.command()
@click.option('--models', '-m', 
              multiple=True,
              type=click.Choice(['LSTM', 'GRU', 'TCN', 'Transformer'], case_sensitive=False),
              help='Model names to train (can specify multiple)')
@click.option('--lag_params', '-l',
              multiple=True,
              type=int,
              help='Lag parameters to test (can specify multiple)')
@click.option('--dataset', '-d',
              multiple=True, 
              type=click.Choice(['Padding', 'Non-padding'], case_sensitive=False),
              help="Use dataset with padding or without (or both)")


def main(models: List[str], lag_params: List[int], dataset: List[str]):
    """Train and evaluate models with different lag parameters.

    if no model is specified, trains all models
    if no lag parameters are specified, uses 5
    
    Examples:
        python main.py -m LSTM -m GRU -l 5 -l 10
        python main.py --models Transformer --default_lags
    """
    # Process lag parameters
    lags = list(lag_params) if lag_params else [10] 

    datasets_to_evaluate = []
    if len(dataset) == 0:
        dataset = ["Padding", "Non-Padding"]
    if "Non-Padding" in dataset:
        datasets_to_evaluate.append(NonPaddingDataset)
    if "Padding" in dataset:
        datasets_to_evaluate.append(PaddingDataset)
    
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
        click.echo(f"\t{model.name}")
    click.echo(f"Using lag parameters: {', '.join(map(str, lags))}")
    click.echo(f"On datasets: {', '.join(map(str, dataset))}")
 


if __name__ == '__main__':
    main()




