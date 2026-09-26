# Kaggle GPU stage

`dense_gpu.py` is the reproducible accelerator stage. It expects the three raw
Parquet files as a private Kaggle dataset, requires two GPUs, encodes fixed
passages with the Russian-focused `deepvk/USER-bge-m3`, performs exact sharded cosine
search, and exports compact item-level rankings for validation and benchmark.

Raw data, model weights and embeddings are not committed. `scripts/fetch_kaggle_output.sh`
downloads only `dense_rankings.parquet`, `validation_labels.parquet`, and run metadata.

`splade/06_splade_gpu_experiment.ipynb` is the next accelerator experiment. It
uses the Russian-specific `naver/neuclir22-splade-ru` checkpoint and exports
wide rankings for controlled query/document pruning, chunking, and local-channel
ablations. Its CC BY-NC-SA 4.0 license is recorded explicitly. Use
`scripts/fetch_splade_output.sh` after the kernel completes.

`finetune/08_finetune_dense_gpu.ipynb` performs leakage-safe LoRA domain
adaptation of `deepvk/USER-bge-m3` on Avito clicks using both T4 GPUs, then
rebuilds filter-aware rankings. It exports rankings and run metadata only; use
`scripts/fetch_finetuned_dense_output.sh` after completion.
