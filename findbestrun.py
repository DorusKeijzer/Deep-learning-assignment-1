import os
import json
from os.path import isfile

startdir = "results"

runs = os.listdir(startdir)
best_results = []

for run in runs:
    raw_results_dir = os.path.join(startdir, run, "raw_test_results.json")
    print(raw_results_dir)
    
    if os.path.isfile(raw_results_dir):
        with open(raw_results_dir) as f:
            results = json.load(f)
            for result in results:
                print(result["score"])
                if len(best_results) < 10:
                    best_results.append(result)
                    best_results.sort(key = lambda x: x["score"])
                elif result["score"] < best_results[-1]["score"]:
                    best_results.pop()
                    best_results.append(result)
                    best_results.sort(key = lambda x: x["score"])

for best_result in best_results:
    for key in best_result.keys():
        print(key, best_result[key])
                    



