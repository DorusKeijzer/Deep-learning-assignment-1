import os
import json
from collections import defaultdict

startdir = "results"

# model_name -> model_parameters -> list of scores
aggregated_scores = defaultdict(lambda: defaultdict(list))

runs = os.listdir(startdir)

for run in runs:
    raw_results_path = os.path.join(startdir, run, "raw_test_results.json")
    
    if os.path.isfile(raw_results_path):
        with open(raw_results_path) as f:
            results = json.load(f)
            for result in results:
                model_name = result["model_name"]
                params = result["model_parameters"]
                score = result["score"]
                aggregated_scores[model_name][params].append(score)

# Print aggregated results
for model_name, param_dict in aggregated_scores.items():
    print(model_name)
    for params, scores in param_dict.items():
        avg_score = sum(scores) / len(scores)
        count = len(scores)
        print(f"\t{params}: score: {avg_score:.4f}, occurrences: {count}")

