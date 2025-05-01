Runs are stored as directories with a timestamp to differentiate them

Each run contains a log of the run and `plots` subdirectory containing timestamped plots and a `weights` directory containing model weights. Gitignore is set to ignore model weights, so model weights are kept locally but not shared to prevent us running out of storage. We can share good model weights amongst ourselves.
