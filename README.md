# Deep Learning Assignment 1

## Poetry
if you have [Poetry](https://python-poetry.org/) installed on your machine, you can run any file using:

```bash
poetry run python <filename>
```

otherwise you have to make sure you have installed all dependencies

## main.py

`main.py` runs the training and evaluation loop for a specified selection of models as defined in `model/architectures/models` for the specified lag features. E.g.

```bash       
python main.py -m LSTM -m GRU -l 5 -l 10
python main.py --models GRU --default_lags

```


