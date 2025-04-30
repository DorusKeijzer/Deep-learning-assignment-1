from model_architectures.baseline import MostRecentBaseline, MeanBaseline
from model_architectures.gru import GRUModel

baselines = [MostRecentBaseline(), MeanBaseline()]

GRUs = [GRUModel(64,1), GRUModel(64,2), GRUModel(64,3), GRUModel(64,4)]

# LSTMs = ... 

# ...

models = baselines + GRUs # + LSTMs


if __name__ == "__main__":
    for model in models:
        print(model.name)
