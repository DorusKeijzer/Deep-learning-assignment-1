from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel

baselines = [(MostRecentBaseline, {}),
             (MeanBaseline, {})]
# model configurations for GRUModel

GRUs = [
    (GRUModel, {"hidden_dim": 64, "num_layers": 1}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 2}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 3}),
    (GRUModel, {"hidden_dim": 64, "num_layers": 4}),
]




# LSTMs = ... 

# ...

models = baselines + GRUs # + LSTMs


if __name__ == "__main__":
    for model in models:
        print(model.name)
