# Kaggle GPU stage

`dense_gpu.py` is the reproducible accelerator stage. It expects the three raw
Parquet files as a private Kaggle dataset, requires two GPUs, encodes fixed
passages with the Russian-focused `deepvk/USER-bge-m3`, performs exact sharded cosine
search, and exports compact item-level rankings for validation and benchmark.

Raw data, model weights and embeddings are not committed. `scripts/fetch_kaggle_output.sh`
downloads only `dense_rankings.parquet`, `validation_labels.parquet`, and run metadata.
