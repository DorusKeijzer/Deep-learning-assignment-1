from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel
from model_architectures.one_dim_cnn import CNN1DModel

baselines = [(MostRecentBaseline, {}),
             (MeanBaseline, {})]
# model configurations for GRUModel:
# hidden_dim: number of hidden dimension

GRUs = [
    (GRUModel, {"hidden_dim": 64, "num_layers": 1}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 2}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 3}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 4}),
]

# 1D CNNs: 
OneDCNNs = [
    (CNN1DModel, {})
]


# LSTMs = ... 

# ...

models = baselines + GRUs + OneDCNNs # + LSTMs


if __name__ == "__main__":
    for model in models:
        print(model.name)
