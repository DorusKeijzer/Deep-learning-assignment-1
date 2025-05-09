import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Raw list from your output
raw_data = """
GRU\thidden dimensions: 128, number of layers: 1\t5\t13.5688
GRU\thidden dimensions: 128, number of layers: 2\t10\t10.7851
GRU\thidden dimensions: 128, number of layers: 3\t15\t12.0332
GRU\thidden dimensions: 128, number of layers: 4\t15\t18.7984
GRU\thidden dimensions: 32, number of layers: 1\t10\t10.4446
GRU\thidden dimensions: 32, number of layers: 2\t10\t9.8092
GRU\thidden dimensions: 32, number of layers: 3\t15\t10.4717
GRU\thidden dimensions: 32, number of layers: 4\t20\t10.1374
GRU\thidden dimensions: 64, number of layers: 1\t5\t12.2013
GRU\thidden dimensions: 64, number of layers: 2\t10\t9.8159
GRU\thidden dimensions: 64, number of layers: 3\t10\t11.7503
GRU\thidden dimensions: 64, number of layers: 4\t10\t10.6217
LSTM\thidden dimensions: 128, number of layers: 3\t11\t5.4882
LSTM\thidden dimensions: 64, number of layers: 1, dropout: 0.2\t18\t9.9883
LSTM\thidden dimensions: 64, number of layers: 2, dropout: 0.2\t18\t10.3153
LSTM\thidden dimensions: 64, number of layers: 3, dropout: 0.2\t18\t13.8475
SimpleRNN\thidden dimensions: 128, number of layers: 1\t35\t26.3969
SimpleRNN\thidden dimensions: 128, number of layers: 2\t25\t25.8078
SimpleRNN\thidden dimensions: 32, number of layers: 1\t15\t18.2126
SimpleRNN\thidden dimensions: 32, number of layers: 2\t15\t15.3582
SimpleRNN\thidden dimensions: 64, number of layers: 1\t20\t17.6871
SimpleRNN\thidden dimensions: 64, number of layers: 2\t15\t17.2216
"""

# Convert to DataFrame
data = [line.split("\t") for line in raw_data.strip().split("\n")]
df = pd.DataFrame(data, columns=["model_name", "model_config", "best_lag", "avg_score"])
df["best_lag"] = df["best_lag"].astype(int)
df["avg_score"] = df["avg_score"].astype(float)

# Combine model name and config for a clearer label
df["label"] = df["model_name"] + " | " + df["model_config"]

# Create pivot table: rows = model, columns = lag, values = score
pivot = df.pivot(index="label", columns="best_lag", values="avg_score")

# Plot heatmap
plt.figure(figsize=(14, len(df) * 0.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Avg Score"})
plt.title("Best Lag and Average Score per Model Configuration")
plt.xlabel("Best Lag")
plt.ylabel("Model Configuration")
plt.tight_layout()
plt.show()

