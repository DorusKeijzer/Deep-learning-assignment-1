from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel, AttentionGRUModel
from model_architectures.one_dim_cnn import CNN1DModel
from model_architectures.tcn import TCNModel
from model_architectures.rnn import SimpleRNNModel
from model_architectures.lstm import LSTMModel
from model_architectures.transformer import TransformerModel
from model_architectures.n_beats import NBEATSModel
from model_architectures.cnn_lstm_hybrid import CNNLSTMModel
from model_architectures.tcn import TCNModel

#
baselines = [
# (MostRecentBaseline, {}),

#              (MeanBaseline, {})
]
# model configurations for GRUModel:
# hidden_dim: number of hidden dimension
#
GRUs = [
#    (GRUModel, {"hidden_dim": 32, "num_layers": 1}),
    # (GRUModel, {"hidden_dim": 32, "num_layers": 2}),
    # (GRUModel, {"hidden_dim": 32, "num_layers": 3}),
    # (GRUModel, {"hidden_dim": 32, "num_layers": 4}),
    # (GRUModel, {"hidden_dim": 64, "num_layers": 1}),    
    # (GRUModel, {"hidden_dim": 64, "num_layers": 2}),
    # (GRUModel, {"hidden_dim": 64, "num_layers": 3}),
    # (GRUModel, {"hidden_dim": 64, "num_layers": 4}),
    # (GRUModel, {"hidden_dim": 128, "num_layers": 1}),
    # (GRUModel, {"hidden_dim": 128, "num_layers": 2}),
    # (GRUModel, {"hidden_dim": 128, "num_layers": 3}),
    # (GRUModel, {"hidden_dim": 128, "num_layers": 4}),

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
    # (CNN1DModel, {"num_channels": [128, 256], "kernel_sizes": [7, 7], "dropout": 0.3, "use_batch_norm": False}),
    # (CNN1DModel, {"num_channels": [64, 128], "kernel_sizes": [3, 3], "stride": 2, "pooling": 'avg', "adaptive_pooling": False}),
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
]


LSTMs = [
    # (LSTMModel, {"hidden_dim": 16, "num_layers": 1}),
    # (LSTMModel, {"hidden_dim": 16, "num_layers": 2}),
    # (LSTMModel, {"hidden_dim": 16, "num_layers": 3}),
    # (LSTMModel, {"hidden_dim": 32, "num_layers": 1}),
    # (LSTMModel, {"hidden_dim": 32, "num_layers": 2}),
    # (LSTMModel, {"hidden_dim": 32, "num_layers": 3}),
    # (LSTMModel, {"hidden_dim": 64, "num_layers": 1}),
    # (LSTMModel, {"hidden_dim": 64, "num_layers": 2}),
    # (LSTMModel, {"hidden_dim": 64, "num_layers": 3}),
    # (LSTMModel, {"hidden_dim": 128, "num_layers": 1}),
    # (LSTMModel, {"hidden_dim": 128, "num_layers": 2}),
    (LSTMModel, {"hidden_dim": 128, "num_layers":   3}),
]

# ...

models = baselines + GRUs + OneDCNNs + TCNs + LSTMs

Transformers = [
    (TransformerModel, {"d_model": 64, "nhead": 4, "num_layers": 2}),
    (TransformerModel, {"d_model": 128, "nhead": 8, "num_layers": 3}),
]

Hybrids = [
    # (CNNLSTMModel, {"cnn_channels": [32, 64], "lstm_hidden": 64}),
    (AttentionGRUModel, {"hidden_dim": 64, "num_layers": 2, "attention_units": 32}),
]

#
# NBEATS = [
#     (NBEATSModel, {"stack_types": ["generic", "seasonal"], "num_blocks": 3}),
# ]
#
models += Transformers + Hybrids + RNNs #+ NBEATS

if __name__ == "__main__":
    for model in models:
        print(model.name)
