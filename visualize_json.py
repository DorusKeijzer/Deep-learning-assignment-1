import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
with open("/home/dorus/Documenten/UU/Blok 4/deep_learning/Deep-learning-assignment-1/results/2025-05-06_18-13-41/raw_test_results.json") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter to one model type (e.g., GRU)
df = df[df["model_name"] == "GRU"]

# Set styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Line Plot: Score vs Lag Param (one line per model_parameters)
plt.figure()
sns.lineplot(data=df, x="lag_param", y="score", hue="model_parameters", marker="o", ci=None)
plt.title("GRU Score vs Lag Parameter (by Model Parameters)")
plt.xlabel("Lag Parameter")
plt.ylabel("Score")
plt.legend(title="GRU Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Facet Grid: One plot per parameter config
g = sns.FacetGrid(df, col="model_parameters", col_wrap=3, sharey=False, height=4)
g.map_dataframe(sns.lineplot, x="lag_param", y="score", marker="o")
g.set_axis_labels("Lag Parameter", "Score")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Score vs Lag per GRU Configuration")
plt.show()

# 3. Boxplot: Score Distribution per Lag Param, Grouped by GRU parameters
plt.figure()
sns.boxplot(data=df, x="lag_param", y="score", hue="model_parameters")
plt.title("Score Distribution by Lag Parameter and GRU Parameters")
plt.xlabel("Lag Parameter")
plt.ylabel("Score")
plt.legend(title="GRU Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Heatmap: GRU Parameters vs Lag Param with Mean Score
pivot_table = df.groupby(["model_parameters", "lag_param"])["score"].mean().unstack()
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Average Score per GRU Configuration (Lower is Better)")
plt.xlabel("Lag Parameter")
plt.ylabel("GRU Parameters")
plt.tight_layout()
plt.show()

# 5. Line Plot with Error Bars: Mean ± Std for Each GRU parameter config
agg = df.groupby(["model_parameters", "lag_param"])["score"].agg(["mean", "std"]).reset_index()
plt.figure()
for param in agg["model_parameters"].unique():
    subset = agg[agg["model_parameters"] == param]
    plt.errorbar(subset["lag_param"], subset["mean"], yerr=subset["std"], label=param, marker='o', capsize=5)

plt.title("GRU Score vs Lag (Mean ± Std) by Parameters")
plt.xlabel("Lag Parameter")
plt.ylabel("Score")
plt.legend(title="GRU Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load data
with open("/home/dorus/Documenten/UU/Blok 4/deep_learning/Deep-learning-assignment-1/results/2025-05-06_18-13-41/raw_test_results.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Filter to one model (e.g., GRU)
df = df[df["model_name"] == "GRU"]

# Extract hidden_dim and num_layers from model_parameters
def parse_params(param_str):
    match = re.search(r"hidden dimensions: (\d+), number of layers: (\d+)", param_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

df[["hidden_dim", "num_layers"]] = df["model_parameters"].apply(lambda x: pd.Series(parse_params(x)))

# Find the lowest score per (hidden_dim, num_layers) combo
best_configs = df.loc[df.groupby(["hidden_dim", "num_layers"])["score"].idxmin()]

# Create pivot tables
score_pivot = best_configs.pivot(index="hidden_dim", columns="num_layers", values="score")
lag_pivot = best_configs.pivot(index="hidden_dim", columns="num_layers", values="lag_param")

# Plot heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(score_pivot, annot=lag_pivot, fmt="", cmap="viridis", cbar_kws={"label": "Lowest Score"})
plt.title("Best GRU Score by Hidden Dim & Num Layers\n(Annotation = Lag Param at Best Score)")
plt.xlabel("Number of Layers")
plt.ylabel("Hidden Dimensions")
plt.tight_layout()
plt.show()

