from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel
from model_architectures.one_dim_cnn import CNN1DModel
from model_architectures.tcn import TCNModel
from model_architectures.rnn import SimpleRNNModel
from model_architectures.lstm import LSTMModel

baselines = [(MostRecentBaseline, {}),
             (MeanBaseline, {})]
# model configurations for GRUModel:
# hidden_dim: number of hidden dimension

GRUs = [
    (GRUModel, {"hidden_dim": 32, "num_layers": 1}),
    (GRUModel, {"hidden_dim": 32, "num_layers": 2}),
    (GRUModel, {"hidden_dim": 32, "num_layers": 3}),
    (GRUModel, {"hidden_dim": 32, "num_layers": 4}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 1}),    
    (GRUModel, {"hidden_dim": 64, "num_layers": 2}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 3}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 4}),
    (GRUModel, {"hidden_dim": 128, "num_layers": 1}),
    (GRUModel, {"hidden_dim": 128, "num_layers": 2}),
    (GRUModel, {"hidden_dim": 128, "num_layers": 3}),
    (GRUModel, {"hidden_dim": 128, "num_layers": 4}),

]

TESTGRU = [
    (GRUModel, {"hidden_dim": 32, "num_layers": 2})
]   


# 1D CNNs: 

OneDCNNs = [
    (CNN1DModel, {"num_channels": [32, 64], "kernel_sizes": [3, 5], "dropout": 0.2, "use_batch_norm": True}),
    (CNN1DModel, {"num_channels": [64, 128], "kernel_sizes": [5, 5], "dropout": 0.3, "use_batch_norm": True}),
    (CNN1DModel, {"num_channels": [32, 32], "kernel_sizes": [3, 3], "dropout": 0.2, "use_batch_norm": False}),
    (CNN1DModel, {"num_channels": [64, 128], "kernel_sizes": [3, 3], "dropout": 0.5, "use_batch_norm": True}),
    (CNN1DModel, {"num_channels": [128, 256], "kernel_sizes": [7, 7], "dropout": 0.3, "use_batch_norm": False}),
    (CNN1DModel, {"num_channels": [64, 128], "kernel_sizes": [3, 3], "stride": 2, "pooling": 'avg', "adaptive_pooling": False}),
]



TCNs = [
    (TCNModel, {"num_channels": [32, 32], "kernel_size": 3, "dropout": 0.2}),
    (TCNModel, {"num_channels": [64, 64], "kernel_size": 3, "dropout": 0.3}),
    (TCNModel, {"num_channels": [32, 64], "kernel_size": 5, "dropout": 0.2}),
]

RNNs = [
    (SimpleRNNModel, {"hidden_dim": 32, "num_layers": 1}),
    (SimpleRNNModel, {"hidden_dim": 32, "num_layers": 2}),
    (SimpleRNNModel, {"hidden_dim": 64, "num_layers": 1}),
    (SimpleRNNModel, {"hidden_dim": 64, "num_layers": 2}),
    (SimpleRNNModel, {"hidden_dim": 128, "num_layers": 1}),
    (SimpleRNNModel, {"hidden_dim": 128, "num_layers": 2}),
    (SimpleRNNModel, {"hidden_dim": 256, "num_layers": 1}),
    (SimpleRNNModel, {"hidden_dim": 256, "num_layers": 2}),
]


LSTMs = [
    (LSTMModel, {"hidden_dim": 16, "num_layers": 1}),
    (LSTMModel, {"hidden_dim": 16, "num_layers": 2}),
    (LSTMModel, {"hidden_dim": 32, "num_layers": 1}),
    (LSTMModel, {"hidden_dim": 32, "num_layers": 2}),
    (LSTMModel, {"hidden_dim": 64, "num_layers": 1}),
    (LSTMModel, {"hidden_dim": 64, "num_layers": 2}),
    (LSTMModel, {"hidden_dim": 128, "num_layers": 1}),
    (LSTMModel, {"hidden_dim": 128, "num_layers": 2}),
]

# ...

models = baselines + GRUs + OneDCNNs + TCNs + LSTMs


if __name__ == "__main__":
    for model in models:
        print(model.name)
