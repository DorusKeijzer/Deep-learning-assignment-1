Define model classes in a seperate file per model, e.g. `LSTM.py` for the LSTM class, `GRU.py` for the GRU class etc.

All of the files in this directory are designed as dependencies to be called from the root (so not from inside this directory)

## models.py
Define instances of models in `models.py` by defininig a model type and a dictionary of parameters:, e.g.: 


```python
GRUs = [
    (GRUModel, {"hidden_dim": 64, "lag": 1}),
    (GRUModel, {"hidden_dim": 64, "lag": 2}),
    (GRUModel, {"hidden_dim": 64, "lag": 3}),
    (GRUModel, {"hidden_dim": 64, "lag": 4}),
]

```
The variable `models` should contain all models.

## base_model.py
Contains a base model class that all models should implement in order to make them run smoothly in the training loop.

To implement the base model you need to define the `forward()` and `name()` methods. Name method should use the `@property` decorator and return a name that differentiates the model from other models and instances of the same model as to account for all possible hyperparameters of the model (except input size as that can be deduced from the lag parameter), e.g. `GRU model, hidden dimensions: 5, number of layers: 32"`, etc.

Finally: the base model takes `input_size` as a parameter, and so should models that inherit from this base class. This input size will correspond to the lag parameter during the training run.

# specific models: 

## Baseline.py
Contains baseline models to compare to as a sanity check: 

-`MostRecentBaseline`: predicts the most recent value 
-`MeanBaseline`: predicts the mean of the feature vector

## GRU.py
https://en.wikipedia.org/wiki/Gated_recurrent_unit
