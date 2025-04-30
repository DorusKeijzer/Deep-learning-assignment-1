from utils.dataloader import PaddingDataset
from model_architectures.models import models as ALL_MODELS
from model_architectures.models import GRUs # LSTMs, etc.
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
@click.option('--default_lags', is_flag=True,
              help='Use default lag parameters [5, 10, 20] instead of specifying')

def main(models: Tuple[str], lag_params: Tuple[int], default_lags: bool):
    """Train and evaluate models with different lag parameters.

    if no model is specified, trains all models
    if no lag parameters are specified, uses 5,10,20
    
    Examples:
        python main.py -m LSTM -m GRU -l 5 -l 10
        python main.py --models Transformer --default_lags
    """
    # Process lag parameters
    if default_lags:
        lags = [5, 10, 20]
    else:
        lags = list(lag_params) if lag_params else [10]  # Default single lag
    
    # Process models
    model_names = list(set(models)) if models else ALL_MODELS
    
    click.echo(f"Training models: {', '.join(model_names)}")
    click.echo(f"Using lag parameters: {', '.join(map(str, lags))}")
    

if __name__ == '__main__':
    main()




