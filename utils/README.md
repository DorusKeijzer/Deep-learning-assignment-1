## dataloader.py

Contains two data loaders, that both have a `lag_size` parameter:

-`NonPaddingDataset`: adds padding to the dataset so that we can make use of the first target variables (i.e. 1000 datapoints every time)
-`PaddingDataset`: adds no padding, so starts at the `lag_size`th entry. (i.e. has 1000-`lag_size` datapoints)


