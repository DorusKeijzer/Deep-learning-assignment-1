import os
import json
import pandas as pd

startdir = "results"
records = []

# Load data
for run in os.listdir(startdir):
    path = os.path.join(startdir, run, "raw_test_results.json")
    if os.path.isfile(path):
        with open(path) as f:
            results = json.load(f)
            records.extend(results)

# Load into DataFrame
df = pd.DataFrame(records)

# Group by model name, parameters, and lag
grouped = df.groupby(["model_name", "model_parameters", "lag_param"])["score"].mean().reset_index()

# Find optimal lag per model configuration
best_lags = grouped.loc[grouped.groupby(["model_name", "model_parameters"])["score"].idxmin()]

# Print
for _, row in best_lags.iterrows():
    print(f"{row['model_name']}\n\t{row['model_parameters']}: best lag = {row['lag_param']}, avg score = {row['score']:.4f}")

