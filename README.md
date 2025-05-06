# Deep Learning Assignment 1

## Poetry
if you have [Poetry](https://python-poetry.org/) installed on your machine, you can run any file using:

```bash
poetry run python <filename>
```

otherwise you have to make sure you have installed all dependencies

## main.py

`main.py` runs the training and evaluation loop for a specified selection of models as defined in `model/architectures/models` for the specified lag features. E.g.

Args:
- `models`: List of models to train. Choices are 'LSTM', 'GRU', 'TCN', or 'Transformer'. If not specified, all models will be trained.
- `lag_params``: List of lag parameters to test. Default is [5] if not specified.
- `datasets`: List of dataset types to use. Choices are 'Padding' or 'Non-padding'. Default is both `['Padding', 'Non-padding']`.
- `epochs`: Number of training epochs. Default is 100.
- `learning_rate`: Learning rate for training. Default is 1e-3.
- `no_early_stopping`: If True, runs all epochs without early stopping. If False (default), may stop early based on validation performance.
- `no_show`: If True, does not show  plots between runs
- `lag_params`: either an integer, or a range of integers (`l 5-10`)

Notes:
  - Multiple models, lag parameters, and datasets can be specified by repeating the flags
  - Model names and dataset types are case-insensitive

Examples: 

```bash       
    python main.py -m LSTM -m GRU -l 5-10
    python main.py --models Transformer --lag_params 5 --dataset Padding
    python main.py -d Padding -d Non-padding --epochs 50 --no-early-stopping
    python main.py

```

for more information: 
```bash       
python main.py --help
```


