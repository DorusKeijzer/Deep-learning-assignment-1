Seperate file per model, e.g. `LSTM.py` for LSTM, `GRU.py` for GRU etc.

`models.py` contains a variable `models` that is a list of these models

# base_model.py
Contains a base model class that all models should implement in order to make them run smoothly in the training loop

# Baseline.py
Contains baseline models to compare to as a sanity check: 

-`MostRecentBaseline`: predicts the most recent value 
-`MeanBaseline`: predicts the mean of the feature vector

# GRU.py
https://en.wikipedia.org/wiki/Gated_recurrent_unit
